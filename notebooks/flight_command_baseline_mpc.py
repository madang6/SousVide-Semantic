import os
import sys, signal, atexit
import curses, builtins
from copy import deepcopy
import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import json
import os
import argparse
from tabulate import tabulate
import time
from enum import Enum

import cv2

import sys, select, termios, tty
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    VehicleOdometry,
    VehicleRatesSetpoint,
    TrajectorySetpoint
)

from sousvide.control.pilot import Pilot
import figs.control.vehicle_rate_mpc as vr_mpc

import sousvide.flight.vision_preprocess_alternate as vp
import sousvide.flight.zed_command_helper as zch
import figs.dynamics.model_equations as me
import figs.dynamics.model_specifications as qc
import figs.utilities.trajectory_helper as th
import figs.tsplines.min_snap as ms
import sousvide.visualize.plot_synthesize as ps
import sousvide.visualize.record_flight as rf
# import sousvide.visualize.plot_flight as pf

class StateMachine(Enum):
    """Enum class for pilot state."""
    INIT   = 0
    READY  = 1
    ACTIVE = 2
    LAND   = 3
    HOLD   = 4
    SPIN   = 5

class FlightCommand(Node):
    """Node for generating commands from SFTI."""

    def __init__(self,mission_name:str) -> None:
        """ Flight Command Node Constructor.
        
        Args:
            mission_name (str): Name of the mission configuration file.
        
        Variables:
            isNN (bool): Flag for neural network controller.
            ctl (Pilot): Controller object.

            Subscribers:
            internal_estimator_subscriber: Internal estimator subscriber.
            external_estimator_subscriber: External estimator subscriber.
            vehicle_rates_subcriber: Vehicle rates subscriber.

            Publishers:
            vehicle_command_publisher: Vehicle command publisher.
            offboard_control_mode_publisher: Offboard control mode publisher.
            vehicle_rates_setpoint_publisher: Vehicle rates setpoint publisher.

            pipeline (rs.pipeline):         Camera pipeline.
            cam_dim (list):                 Camera dimensions.
            cam_fps (int):                  Camera frames per second.
            node_start (bool):              Node start flag.
            node_status_informed (bool):    Node status informed flag.
            Tpi (np.ndarray):               Trajectory time.
            CPi (np.ndarray):               Control points.
            drone_config (dict):            Drone configuration.
            obj (np.ndarray):               Objective.
            k_sm (int):                     Current segment index.
            N_sm (int):                     Total number of segments.
            k_rdy (int):                    Ready counter.
            t_tr0 (float):                  Trajectory start time.
            vo_est (VehicleOdometry):       Current state vector (estimated).
            vo_ext (VehicleOdometry):       Current state vector (external).
            vrs (VehicleRatesSetpoint):     Current vehicle rates setpoint.
            znn (torch.Tensor):             Current visual feature vector.
            sm (StateMachine):              State machine.
            q0 (np.ndarray):                Initial attitude.
            z0 (float):                     Initial altitude.
            q0_tol (float):                 Attitude tolerance.
            z0_tol (float):                 Altitude tolerance.
            t_lg (float):                   Linger time.
            recorder (FlightRecorder):      Flight recorder object.
        """
        super().__init__('outdoor_command_node')

#NOTE Keyboard stuff
        # (optional? - for the non-curses implementation)
        # Save and configure terminal for raw (unbuffered) input
        self._orig_tty = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())      # Make reads return immediately
        
        self.key_pressed = None
        self.kb_shutdown = threading.Event()
        self.kb_thread = threading.Thread(target=self.kb_loop, daemon=True)
        self.kb_thread.start()
#
        print("=====================================================================")
        print("---------------------------------------------------------------------")

        # ---------------------------------------------------------------------------------------
        # Some useful constants & variables -----------------------------------------------------
        # ---------------------------------------------------------------------------------------
#TODO should this really be 20 or should it be 10?
        hz = 20
#
        # Camera Parameters
        cam_dim,cam_fps, = [376,672,3],30           # we use the convention: [height,width,channels]

        # QoS Profile
        qos_drone = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_mocap = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        print("Loading Configs")
        # Load Configs
        workspace_path = os.path.dirname(os.path.dirname(__file__))

        # Check for mission file
        if os.path.isfile(os.path.join(workspace_path,"configs","missions",mission_name+".json")):
            with open(os.path.join(workspace_path,"configs","missions",mission_name+".json")) as json_file:
                mission_config = json.load(json_file)
        else:
            raise FileNotFoundError(f"Mission file not found: {mission_name}.json")
        print("Loading Courses")

        # Check for trajectory file, skip if it doesn't exist
        loaded_tXUi = None
        if mission_config.get("course") is not None:
            with open(os.path.join(workspace_path,"configs","scenes",mission_config["course"]+"_tXUi.pkl"),"rb") as f:
                loaded_tXUi = pickle.load(f)
        else:
            print("No Courses Loaded")

        # Check for drone file
        if os.path.isfile(os.path.join(workspace_path,"configs","drones",mission_config["frame"]+".json")):
            with open(os.path.join(workspace_path,"configs","drones",mission_config["frame"]+".json")) as json_file:
                frame_config = json.load(json_file)
        else:
            raise FileNotFoundError(f"Drone frame file not found, please provide one")
        print("Loaded Frame")
        if loaded_tXUi is not None:
            tXUi = deepcopy(loaded_tXUi)

        # Unpack drone
        drone_config = qc.generate_specifications(frame_config)
        drone_prefix = '/'+mission_config["drone"]

        # Unpack variables
        q0_tol,z0_tol = mission_config["failsafes"]["attitude_tolerance"],mission_config["failsafes"]["height_tolerance"]
        self.t_lg = mission_config["linger"]

        # ---------------------------------------------------------------------------------------
        # Class Variables -----------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        print("Loading Pilot")
        # Controller
        if mission_config["pilot"] == "mpc":
            self.isNN = False
            if loaded_tXUi is None:
                raise RuntimeError(
                    "Pilot 'mpc' requires a precomputed tXUi trajectory, but no "
                    f"'{mission_config['course']}_tXUi.pkl' was found."
                )
            self.ctl = vr_mpc.VehicleRateMPC(tXUi,drone_config,hz,use_RTI=True,tpad=mission_config["linger"])
        elif mission_config["pilot"] == "range_velocity":
            self.isNN = False
            self.use_depth = True
            # range controller tunables
            self.range_target_m   = float(mission_config.get('range_target_m', 2.0))
            self.range_top_pct    = float(mission_config.get('range_top_percent', 10.0))  # top-P% similarity pixels
            self.range_min_pixels  = int(mission_config.get("range_min_pixels", 200))
            self.range_standoff   = float(mission_config.get('range_standoff', 0.5))  # standoff distance in meters
            self.range_kp         = float(mission_config.get('range_kp', 0.6))
            self.range_vmax       = float(mission_config.get('range_vmax', 0.05))
            self.range_deadband   = float(mission_config.get('range_deadband', 0.05))
            self.range_k_rate     = float(mission_config.get('range_k_rate', 0.5))
            self.range_max_rate   = float(mission_config.get('range_max_rate', 0.3))
        else:
            self.isNN = True
            self.use_depth = False
            self.ctl = Pilot(mission_config["cohort"],mission_config["pilot"])
            self.ctl.set_mode('deploy')

        # Create subscribers
        self.internal_estimator_subscriber = self.create_subscription(
            VehicleOdometry, drone_prefix+'/fmu/out/vehicle_odometry', self.internal_estimator_callback, qos_drone)
        self.external_estimator_subscriber = self.create_subscription(
            VehicleOdometry, '/drone4/fmu/in/vehicle_visual_odometry', self.external_estimator_callback, qos_mocap)
        self.vehicle_rates_subcriber = self.create_subscription(
            VehicleRatesSetpoint, drone_prefix+'/fmu/out/vehicle_rates_setpoint', self.vehicle_rates_setpoint_callback, qos_drone)

        # Create publishers
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand,drone_prefix+'/fmu/in/vehicle_command', qos_drone)
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode,drone_prefix+'/fmu/in/offboard_control_mode', qos_drone)
        self.vehicle_rates_setpoint_publisher = self.create_publisher(VehicleRatesSetpoint, drone_prefix+'/fmu/in/vehicle_rates_setpoint', qos_drone)
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, drone_prefix+'/fmu/in/trajectory_setpoint', qos_drone)

        # Initialize CLIPSeg Model
        self.prompt = mission_config.get('prompt', '')
        print(f"Query Set to: {self.prompt}")
#FIXME
        self.hold_prompt           = self.prompt
        self.prompt_2              = "mannequin in a shirt"  # TODO: remove this, it's just for testing
        self.active_arm            = False
#
        self.hf_model = mission_config.get('hf_model', 'CIDAS/clipseg-rd64-refined')
        self.onnx_model_path_raw = mission_config.get('onnx_model_path')
        self.onnx_model_path = os.path.expanduser(self.onnx_model_path_raw) if self.onnx_model_path_raw else None

        if self.onnx_model_path is None:
            print("Initializing CLIPSegHFModel...")
            self.vision_model = vp.CLIPSegHFModel(hf_model=self.hf_model)
        else:
            print("Initializing ONNX CLIPSegHFModel (this may export ONNX)...")
            self.vision_model = vp.CLIPSegHFModel(
                hf_model=self.hf_model,
                onnx_model_path=self.onnx_model_path,
                onnx_model_fp16_path=mission_config.get('onnx_model_fp16_path', None)
        )

        # Initalize camera (if exists)
        self.pipeline = zch.get_camera(cam_dim[0],cam_dim[1],cam_fps,use_depth=self.use_depth)
        if self.pipeline is not None:
            self.cam_dim,self.cam_fps = cam_dim,cam_fps
        else:
            self.cam_dim,self.cam_fps = [0,0,0],0
        self.img_times = []

#async vision loop
        self.vision_thread     = None
        self.vision_shutdown   = False
        self.vision_started    = False
        self.latest_frame      = None
        self.latest_loiter_overlay    = None
        self.latest_mask       = None
        self.latest_similarity = None
        self.latest_depth_xyz  = None
#has_one_large_high_sim_region
        self.finding_thread     = None
        self.finding_shutdown   = False
        self.finding_started    = False
        self.found              = False
        self.sim_score          = 0.0
        self.area_frac          = 0.0
#policy switch flag
        self.ready_active     = False
        self.spin_cycle       = False

    #NOTE streamline testing non-mpc pilots
        # non-MPC (neural) case: give everything a neutral default
        self.dt   = 1.0 / hz                                     # hz (control rate)
        self.Tpi  = np.arange(0.0, 1.0 + self.dt, self.dt)
        self.obj  = np.zeros((18,1))
        self.q0   = np.array([0.0, 0.0, 0.70710678, 0.70710678])          # determines starting orientation for ACTIVE mode to launch
        self.z0   = -0.50                                   # init altitude offset
        self.xref = np.zeros(10)                          # no reference state
        self.uref = -6.90*np.ones((4,))                   # reference input
    #FIXME
        self.alt_hold          = float('nan')
        self.trajectory        = None
        self.trajectory_length = None
        self.query_p_cam       = None

        self.yaw_rate_cmd   = 0.0
        self.kp_yaw         = 0.5                 # gentle
        self.max_yaw_rate   = 0.30                 # rad/s
        self.yaw_deadband   = 10.0                  # ±10°
        self.max_yaw_accel  = 0.60                  # rad/s^2
        self._yawspeed_prev = 0.0
    #
        record_duration = self.Tpi[-1] + self.t_lg        # 

        # Initialize control variables
        self.node_start = False                                 # Node start flag
        self.node_status_informed = False                       # Node status informed flag

        self.drone_config = drone_config                        # drone configuration
        self.k_rdy = 0                                          # ready counter
        self.t_tr0 = 0.0                                        # trajectory start time
        self.vo_est = VehicleOdometry()                         # current state vector (estimated)
        self.vo_ext = VehicleOdometry()                         # current state vector (external)
        self.vrs = VehicleRatesSetpoint()                       # current vehicle rates setpoint

        # State Machine and Failsafe variables
        self.sm = StateMachine.INIT                                    # state machine
        
        # self.q0,self.z0 = tXUi[7:11,0],tXUi[3,0]                       # initial attitude and altitude
        self.q0_tol,self.z0_tol = q0_tol,z0_tol                        # attitude and altitude tolerances

        # Create Variables for logging.
        self.recorder = rf.FlightRecorder(
            drone_config["nx"],
            drone_config["nu"],
            hz,
            record_duration,
            cam_dim,
            self.obj,
            mission_config["cohort"],
            mission_config["scene"],
            mission_config["pilot"],
            Nimps=mission_config["images_per_second"]
        )

        self.cohort_name = mission_config["cohort"]
        self.scene_name = mission_config["scene"]
        self.pilot_name = mission_config["pilot"]

        # Create a timer to publish control commands
        self.cmdLoop = self.create_timer(self.dt, self.commander)

        # Camera to body 
        self.T_c2b = np.array(self.drone_config["T_c2b"], dtype=np.float64)

        # Earth to World Pose
        self.T_e2w = np.eye(4)

        # Print Diagnostics
        print('=====================================================================')
        print("========================= SFTI COMMAND NODE =========================")
        print("---------------------------------------------------------------------")
        print('-----------------------> Node Configuration <------------------------')
        print('Cohort Name      :',mission_config["cohort"])
        print('Scene Name      :',mission_config["scene"])
        print('Pilot Name       :',mission_config["pilot"])
        print('Quad Name        :',drone_config["name"])
        print('Control Frequency:',hz)
        print('Images Per Second:',mission_config["images_per_second"])
        print('Failsafe Tolerances (q_init,z_init):',mission_config["failsafes"]["attitude_tolerance"],mission_config["failsafes"]["height_tolerance"])
        print('=====================================================================')
        print('--------------------> Trajectory Information <-----------------------')
        print('---------------------------------------------------------------------')
        print('Starting Location & Orientation:')
        x0_est = zch.vo2x(self.vo_est,self.T_e2w)[0:10]
        x0_est[6:10] = th.obedient_quaternion(x0_est[6:10],self.xref[6:10])
        print('xref: ',x0_est[1:3])
        print('qref: ',x0_est[6:10])
        print('---------------------------------------------------------------------')
        print('=====================================================================')

        input("Press Enter to proceed...")

#FIXME
    def _cleanup(self):
        # Stop & join threads
        if getattr(self, 'vision_thread', None) and self.vision_started:
            print("Closing Vision Thread")
            self.vision_shutdown = True
            self.vision_thread.join()
            self.vision_started = False
            print("Vision Thread Offline")

        if getattr(self, 'finding_thread', None) and self.finding_started:
            print("Closing Object Detection Thread")
            self.finding_shutdown = True
            self.finding_thread.join()
            self.finding_started = False
            print("Object Detection Thread Offline")

        if getattr(self, 'kb_thread', None):
            print("Closing Keyboard Thread")
            self.kb_shutdown.set()
            self.kb_thread.join()
            print("Keyboard Thread Offline")

        # Camera, controller
        try:
            if self.pipeline:
                zch.close_camera(self.pipeline)
            # if not self.isNN:
            #     try:
            #         self.ctl.clear_generated_code()
            #     except Exception as e:
            #         print(f"Error during controller cleanup: {e}. This is OK in this script")
        except Exception as e:
            print(f"Error during device cleanup: {e}")

        # Tear down ROS node & context
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_tty)
        except Exception as e:
            print(f"Terminal Restoration Bug: {e}")

    def kb_loop(self):
        """Runs in background, grabs single chars without blocking."""
        while rclpy.ok() and not self.kb_shutdown.is_set():
            # wait up to 0.1s for a key
            if select.select([sys.stdin], [], [], 0.1)[0]:
                self.key_pressed = sys.stdin.read(1)
#
    def internal_estimator_callback(self, vehicle_odometry:VehicleOdometry):
        """Callback function for internal vehicle_odometry topic subscriber."""
        self.vo_est = vehicle_odometry

    def external_estimator_callback(self, vehicle_odometry:VehicleOdometry):
        """Callback function for external vehicle_odometry topic subscriber."""
        self.vo_ext = vehicle_odometry

    def vehicle_rates_setpoint_callback(self, vehicle_rates_setpoint:VehicleRatesSetpoint):
        """Callback function for vehicle_rates topic subscriber."""
        self.vrs = vehicle_rates_setpoint

    def get_current_timestamp_time(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(self.get_clock().now().nanoseconds/1e6)
    
    def get_current_trajectory_time(self) -> float:
        """Get current trajectory time."""
        return self.get_clock().now().nanoseconds/1e9 - self.t_tr0

    def async_vision_and_query_loop(self):
        """Continuously grab + infer, store only the newest mask.
           Continuously create straight-line trajectories to the query."""
        while not self.vision_shutdown:
            # grab next image
            imgz, depth_xyz, _, _ = zch.get_image(self.pipeline, use_depth=self.use_depth)
            if imgz is None:
                time.sleep(0.01)
                continue
            t0_img = time.time()
            frame = cv2.cvtColor(imgz, cv2.COLOR_BGR2RGB)
            # process next image with CLIPSeg
            mask, raw_similarity = self.vision_model.clipseg_hf_inference(
                frame,
                self.prompt,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False
            )
            self.latest_frame = frame
            self.latest_depth_xyz = depth_xyz
            self.latest_mask = mask
            self.latest_similarity = raw_similarity

            ok_pose, p_cam, _, _ = zch.pose_from_similarity_xyz(
                self.latest_similarity, self.latest_depth_xyz,
                top_percent=self.range_top_pct,
                min_pixels=self.range_min_pixels
            )
            if ok_pose:
                self.query_p_cam = p_cam

            t_elap = time.time() - t0_img
            self.img_times.append(t_elap)
        
    def async_loiter_calibrate(self):
        while not self.finding_shutdown:
            if not self.spin_cycle:
                time.sleep(0.02)
                continue
            else:
                self.found, self.sim_score, self.area_frac, self.latest_loiter_overlay = self.vision_model.loiter_calibrate(
                logits=self.latest_similarity,           # your logits/similarity map
                frame_img=self.latest_frame,         # original frame in BGR
                active_arm = self.active_arm
                )
                # self.found, self.sim_score, self.area_frac = \
                #     self.vision_model.loiter_calibrate(
                #         logits=self.latest_similarity,
                #         active_arm=self.active_arm
                #     )
                time.sleep(0.02)

    def quat_to_yaw(self,q):
        x, y, z, w = q
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    # def build_yaw_qp(self, xyz, yaw0, t0):
    #     """Background build of the spin trajectory."""
    #     cfg = th.generate_spin_keyframes(
    #         name=f"loiter_spin_{t0:.2f}",
    #         Nco=6,
    #         xyz=xyz,
    #         theta0=yaw0, theta1=yaw0,
    #         time=5.0
    #     )
    #     out = ms.solve(cfg)
    #     if out is False:
    #         print(f"Spin QP failed at t0={t0:.2f}")
    #         return
    #     self.Tps_spin, self.CPs_spin = out
    #     self.tXUd_spin       = th.TS_to_tXU(self.Tps_spin, self.CPs_spin,
    #                                         self.drone_config,
    #                                         self.ctl.hz)
    #     # self.spin_start_time = t0
    #     self.spin_ready     = True
    #     print("Spin QP ready")

    def commander(self) -> None:
        """Main control loop for generating commands."""
        # Looping Callback Actions
        x_est,u_pr = zch.vo2x(self.vo_est,self.T_e2w)[0:10],zch.vrs2uvr(self.vrs)
        x_ext,obj = zch.vo2x(self.vo_ext)[0:10],self.obj
        x_est[6:10] = th.obedient_quaternion(x_est[6:10],self.xref[6:10])
        x_ext[6:10] = th.obedient_quaternion(x_ext[6:10],self.xref[6:10])

        frame = self.latest_frame
        # img = self.latest_mask
        # similarity = self.latest_similarity
        # depth = self.latest_depth_xyz
        loiter_overlay = self.latest_loiter_overlay

        # zch.heartbeat_offboard_control_mode(self.get_current_timestamp_time(),self.offboard_control_mode_publisher)
        if self.sm == StateMachine.ACTIVE:
            # Active/Spin: use body-rate control
            if self.pilot_name == 'range_velocity':
                zch.heartbeat_offboard_control_mode(
                    self.get_current_timestamp_time(),
                    self.offboard_control_mode_publisher,
                    body_rate=False, velocity=True
                )
            else:
                zch.heartbeat_offboard_control_mode(
                    self.get_current_timestamp_time(),
                    self.offboard_control_mode_publisher,
                    body_rate=True, velocity=False
                )
        elif self.sm == StateMachine.HOLD:
            # Hold: use velocity control for smooth hover
            zch.heartbeat_offboard_control_mode(
                self.get_current_timestamp_time(),
                self.offboard_control_mode_publisher,
                body_rate=False,
                velocity=True
            )
        elif self.sm == StateMachine.SPIN:
            # for yaw-in-place: hold position via zero velocity, spin via body_rate
            zch.heartbeat_offboard_control_mode(
                self.get_current_timestamp_time(), self.offboard_control_mode_publisher,
                body_rate=True,  # drive the yaw-rate loop
                velocity=True   # drive the XY/Z velocity loops
            )
    
        else:
            # INIT, READY, LAND, etc.—default to body-rate so commands don’t break
            zch.heartbeat_offboard_control_mode(
                self.get_current_timestamp_time(),
                self.offboard_control_mode_publisher,
                body_rate=True,
                velocity=False
            )
    #FIXME        
        if self.key_pressed == '\x1b':     # ESC
            print("ESC detected, landing…")
            self.sm = StateMachine.LAND
            self.key_pressed = None        # reset
        elif self.key_pressed == 'h':      # H key
            print("H key detected, switching to HOLD mode…")
            self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
            zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            self.sm = StateMachine.HOLD
            self.key_pressed = None        # reset
        elif self.key_pressed == 's':      # s key
            print("S key detected, switching to SPIN mode…")
            self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
            zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            self.sm = StateMachine.SPIN
            self.key_pressed = None        # reset
        elif self.key_pressed == 'q':      # q key
            print("Q key detected, switching to HOLD startup mode…")
            self.k_rdy = 0                                                  # Reset ready counter
            self.ready_active = False                                       # Reset ready active flag
            self.spin_cycle   = True
            self.found = False
            self.finding_shutdown = False
            self.last_print_time = 0.0
            self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
            zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            self.sm = StateMachine.HOLD
            self.key_pressed = None        # reset
    #
        # State Machine
        if self.sm == StateMachine.INIT:
            # Looping State Actions
            # Check if we have state information
            if self.vo_est.timestamp > 0:
                # State Actions
                self.T_e2w[0:3,0:3] = R.from_quat(zch.vo2x(self.vo_est)[6:10]).as_matrix().T

                self.k_rdy = 0
                self.sm = StateMachine.READY
                
                print("Starting Zed Vision Thread")
                if not self.vision_started and self.pipeline is not None:
                    self.vision_shutdown = False
                    self.vision_thread = threading.Thread(
                        target=self.async_vision_and_query_loop,
                        daemon=True,
                        name="ZedVisionThread"
                    )
                    self.vision_thread.start()
                    self.vision_started = True
                print("Vision Thread Online")

                print("Starting Object Detection Thread")
                if not self.finding_started:
                    self.finding_thread = threading.Thread(
                        target=self.async_loiter_calibrate,
                        daemon=True,
                        name="ObjectDetectionThread"
                    )
                    self.finding_thread.start()
                    self.finding_started = True
                print("Object Detection Thread Online")

                # Print State Information
                print('---------------------------------------------------------------------')
                print('Node is ready. Starting Initial Scan.')
                print('---------------------------------------------------------------------')
            else:
                # Looping State Actions
                print('Waiting for state information...')

        elif self.sm == StateMachine.READY:
            # Check if we are ready to start (matching attitude and altitude)
            if np.linalg.norm(x_est[6:10]-self.q0) < self.q0_tol and np.abs(x_est[2]-self.z0) < self.z0_tol:
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
                self.k_rdy = 0                                                  # Reset ready counter
                zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
                self.sm = StateMachine.ACTIVE
                
                # Print State Information
                print('=====================================================================')
                print('Trajectory Started.')
            else:
                # Looping State Actions
                self.k_rdy += 1
                if self.k_rdy % 10 == 0:
                    z0ds,z0cr = np.around(self.z0,2),np.around(x_est[2],2)
                    q0ds,q0cr = np.around(self.q0,2),np.around(x_est[6:10],2)

                    print(f"Desired/Current Altitude+Attitude: ({z0ds}/{z0cr})({q0ds}/{q0cr})")
                
        elif self.sm == StateMachine.ACTIVE:
            # Looping State Actions
            t0_lp  = time.time()                                                # Algorithm start time
            t_tr = self.get_current_trajectory_time()                           # Current trajectory time

            # Convert current query point (camera frame) into world coordinates
            p_body_h = zch.pose_c2b(self.T_c2b, self.query_p_cam)  # homogeneous [x,y,z,1]
            p_world_tgt, _, _ = zch.pose_b2w(x_est, p_body_h)      # your existing helper
            dist_to_query = float(np.linalg.norm(p_world_tgt - x_est[0:3]))

            # Check if we are still in the trajectory
            if (t_tr <= (self.Tpi[-1]+self.t_lg) and (dist_to_query > self.range_target_m)):
                zch.publish_position_setpoint_with_yawspeed(
                    self.get_current_timestamp_time(),
                    self.alt_hold if not np.isnan(self.alt_hold) else x_est[2],
                    self.trajectory[1] if len(self.trajectory) > 1 else self.trajectory[0],
                    self.trajectory_setpoint_publisher,
                    self.yaw_rate_cmd
                )
                # if ok_pose:
                # --- Compute yaw-rate to center target on body x-axis ---
                p_body = zch.pose_c2b(self.T_c2b, self.query_p_cam)[:3]

            #NOTE here's the yaw rate proportional controller
                self.yaw_rate_cmd = 0.0
                if p_body[0] > 0.10:  # front-only gate
                    yaw_err = float(np.arctan2(p_body[1], p_body[0]))
                    if abs(yaw_err) > self.yaw_deadband:
                        y_des  = float(np.clip(self.kp_yaw * yaw_err, -self.max_yaw_rate, self.max_yaw_rate))
                        dy_max = self.max_yaw_accel * self.dt
                        yaw_rate_cmd = float(
                            np.clip(y_des - self._yawspeed_prev, -dy_max, dy_max) + self._yawspeed_prev
                        )
                        self.yaw_rate_cmd = yaw_rate_cmd
            #
                # --- Replan short horizon (straight line, speed-limited) ---
                trajectory, dbg = zch.traj_from_target_pose(
                    x_ext=x_est.copy(),
                    p_cam=self.query_p_cam,
                    T_c2b=self.T_c2b,
                    dt=self.dt,
                    total_time=0.6,                 # short horizon; disposable every tick
                    v_max=self.range_vmax,
                    standoff_m=0.5
                )
                self.trajectory = trajectory
            else:
                # trajectory has ended → HOLD Position
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9                       # Record start time
                self.policy_duration       = t_tr
                self.hold_prompt           = self.prompt
            #FIXME uncomment for multi-obj, comment out for single
                # self.prompt                = self.prompt_2
                # self.vision_model.running_min = float('inf')
                # self.vision_model.running_max = float('-inf')
                # self.vision_model._max_prob_logit = float('-inf')
            #
                # self.hold_state = x_est.copy()
                # self.finding_shutdown = False
            #NOTE True for multi-obj, False for single
                # self.spin_cycle = True
                self.spin_cycle = False                                                
            #
                self.ready_active = False                                                 # Reset ready active flag
                self.found = False
                self.sm                    = StateMachine.HOLD
                print('Trajectory Finished → HOLDing Position, readying for next query...')
            self.recorder.record(loiter_overlay)

        elif self.sm == StateMachine.HOLD:
            t_tr = self.get_current_trajectory_time()                           # Current trajectory time
            zch.publish_velocity_hold(
                self.get_current_timestamp_time(),
                self.trajectory_setpoint_publisher
            )
        #NOTE THIS SHOULD BE REPLACED WITH A NEWER VERSION OF THE READY STATE, BUT FOR NOW IT SUFFICES
            if (self.ready_active is True and self.spin_cycle is False) and t_tr > 3.0:
                print("Detector Thread Sleeping; Switching to Active")
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
                self.ready_active = False                                       # Reset ready active flag
                self.active_arm   = False
                self.found = False
                self.last_print_time = 0.0

                self.alt_hold = x_est[2] if np.isnan(self.alt_hold) else self.alt_hold
                zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)

                self.sm = StateMachine.ACTIVE
                print('=====================================================================')
                print('ACTIVE Started.')
            elif (self.ready_active is False and self.spin_cycle is True) and t_tr > 4.0:
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
                zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)

                self.sm = StateMachine.SPIN
                print('=====================================================================')
                print('SPIN Started.')
            elif self.pilot_name == "range_velocity" and t_tr < 6.0 and (self.active_arm is True and self.spin_cycle is False):
                current_state = x_est.copy()
                # One-time target solve while HOLDing
                # ok_pose, p_cam, _, _ = zch.pose_from_similarity_xyz(
                #     similarity, depth,
                #     top_percent=self.range_top_pct,
                #     min_pixels=self.range_min_pixels
                # )

                # if not ok_pose:
                #     print("[HOLD→TARGET] No valid pose yet (insufficient ROI/depth).")
                #     return

                # # cam -> body -> world
                # T_cb = np.array(self.drone_config["T_c2b"], dtype=np.float64)

                trajectory, dbg = zch.traj_from_target_pose(
                    x_ext=current_state,
                    p_cam=self.query_p_cam,
                    T_c2b=self.T_c2b,
                    dt=self.dt,
                    total_time=2.0,
                    v_max=self.range_vmax,
                    standoff_m=0.5
                )
                self.trajectory         = trajectory
                self.trajectory_length  = trajectory.shape[0]
                self.ready_active       = True

                p_body = zch.pose_c2b(self.T_c2b, self.query_p_cam)[:3]
                self.yaw_rate_cmd = 0.0
                if p_body[0] > 0.10:  # front-only gate
                    yaw_err = float(np.arctan2(p_body[1], p_body[0]))
                    if abs(yaw_err) > self.yaw_deadband:
                        y_des  = float(np.clip(self.kp_yaw * yaw_err, -self.max_yaw_rate, self.max_yaw_rate))
                        dy_max = self.max_yaw_accel * self.dt
                        yaw_rate_cmd = float(
                            np.clip(y_des - self._yawspeed_prev, -dy_max, dy_max) + self._yawspeed_prev
                        )
                        self.yaw_rate_cmd = yaw_rate_cmd

                now = time.time()
                if now - self.last_print_time >= 0.5:
                    print(
                        "[HOLD→TARGET] "
                        f"drone={np.asarray(current_state[0:3]).round(3)}  "
                        f"cam={np.asarray(dbg['p_cam']).round(3)}  "
                        f"body={np.asarray(dbg['p_body']).round(3)}  "
                        f"world_target={np.asarray(dbg['p_world_tgt']).round(3)}  "
                        f"world_goal={np.asarray(dbg['p_world_goal']).round(3)}\n"
                        f"dist={dbg['dist']:.3f} m  steps={dbg['steps']}  "
                        # f"dt={dbg.get('dt',dt):.3f}s  vmax={dbg.get('v_max',vmax):.2f} m/s  "
                        # f"step_cap={dbg['step_cap']:.3f} m  standoff={dbg.get('standoff_m',standoff_m):.2f} m"
                    )
                    self.last_print_time = now
            elif t_tr < 6.0:
                return
            else:
                self.sm = StateMachine.LAND
                print('Trajectory Finished → Landing...')
            self.recorder.record(loiter_overlay)

        elif self.sm == StateMachine.SPIN:
            t_tr = self.get_current_trajectory_time()

            zch.publish_velocity_hold_with_yaw_rate(
                self.get_current_timestamp_time(),
                self.trajectory_setpoint_publisher,
                0.75
                )
            if t_tr < 7.0:
                self.active_arm = False
                # self.recorder.record(img)
            elif t_tr >= 7.0 and t_tr < 14.0:
                self.active_arm = True
                # self.recorder.record(vp.colorize_mask_fast((similarity*255).astype(np.uint8),self.vision_model.lut))
                if self.found:
                    self.t_tr0 = self.get_clock().now().nanoseconds/1e9

                    print(f"largest area={self.vision_model.loiter_area_frac*100:.1f}% "
                    f", best_scoring_area={self.vision_model.loiter_max:.3f} " 
                    f", sim_score={self.sim_score:.3f} "
                    f", area_frac={self.area_frac*100:.1f}% "
                    f", sim_score_diff={self.sim_score-self.vision_model.loiter_max:.3f} "
                    f", area_frac_diff={(self.vision_model.loiter_area_frac-self.area_frac)*100:.1f}%")

                    self.sim_score = 0.0
                    self.area_frac = 0.0
                    self.found = False
                    # self.active_arm = True
                    self.vision_model.loiter_max = 0.0
                    self.vision_model.loiter_area_frac = 0.0
                    
                    # self.ready_active = True
                    self.spin_cycle = False
                    self.sm                    = StateMachine.HOLD
                    print(f'Query {self.prompt} Found → HOLDing Position')

                # self.recorder.record(img)
            else:
                self.t_tr0 = self.get_clock().now().nanoseconds / 1e9

                print(f"largest area={self.vision_model.loiter_area_frac*100:.1f}% "
                f", best_scoring_area={self.vision_model.loiter_max:.3f} " 
                f", sim_score={self.sim_score:.3f} "
                f", area_frac={self.area_frac*100:.1f}% "
                f", sim_score_diff={self.vision_model.loiter_max-self.sim_score:.3f} "
                f", area_frac_diff={(self.vision_model.loiter_area_frac-self.area_frac)*100:.1f}%")

                self.active_arm = False
                self.vision_model.loiter_max = 0.0
                self.vision_model.loiter_area_frac = 0.0

                self.ready_active = False
                self.spin_cycle = False
                self.sm   = StateMachine.HOLD
                print(f'Query {self.prompt} Not Found → HOLDing Position, Preparing to Land')
                # self.recorder.record(img)
            self.recorder.record(loiter_overlay)
        else:
            # State Actions
            zch.land(self.get_current_timestamp_time(),self.vehicle_command_publisher)
                
            # Save data
            output_path = self.recorder.save()

            # Print statistics
            if output_path is not None:
                print('Trajectory Finished.')
                print('=====================================================================')
                
                # Compute and print statistics
            #FIXME
                Tsol = self.recorder.Tsol                  # shape (Ntsol, k)
                cmd_times = Tsol[-1, :]                     # total-loop times
                valid = cmd_times > 0                       # True where ACTIVE wrote something

                # only solver-component rows, only ACTIVE columns
                Tcmp = Tsol[:-1, valid]                     # shape (Ntsol–1, n_active)
                Tcmd = cmd_times[valid]                     # shape (n_active,)
            #    

                Tcmp = Tcmp[~np.all(Tcmp == 0, axis=1)]
                Tcmp_tot = np.sum(Tcmp,axis=0)
                mu_t_cmp = np.mean(Tcmp, axis=1)

                hz_cmp = 1/np.mean(Tcmp_tot)
                hz_cmd = 1/np.mean(Tcmd)
                hz_min = 1/np.max(Tcmd)

                print("Controller Computation Time Statistics")
                print(f"Policy was active for {self.policy_duration:.2f} seconds.")
                print("Policy Component Compute Time (s): ", mu_t_cmp)
                print("Policy Total Compute Frequency (hz):", hz_cmp)
                print("Total Command Loop Frequency (hz): ", hz_cmd)
                print("Controller Minimum Frequency (hz): ", hz_min)
                print("CLIPSeg Inference Statistics")
                print(f"Average CLIPSeg Inference Time (s): {np.mean(self.img_times):.4f}")
                print(f"CLIPSeg frequency (hz): {1/np.mean(self.img_times):.2f}")
                print('=====================================================================')
            
            # Wait for GCS clearance
            self._cleanup()
            input("Press Enter to close node...")
            print("Closing node...")
            self.cmdLoop.cancel()
            exit()

def main() -> None:
    print('Starting flight command node...')

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Flight Command Node.")

    # Add arguments
    parser.add_argument(
        '--mission',
        type=str,
        required=True,
        help='Mission config name (without .json)'
    )

    # Parse the command line arguments
    args = parser.parse_args()

    rclpy.init()
    controller = FlightCommand(args.mission)
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

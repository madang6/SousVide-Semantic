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
    VehicleAttitudeSetpoint,
    VehicleLocalPosition,
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
    TEST   = 6

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
        self.local_position_estimator_subscriber = self.create_subscription(
            VehicleLocalPosition, drone_prefix+'/fmu/out/vehicle_local_position', self.external_estimator_callback, qos_drone)
        self.vehicle_rates_subcriber = self.create_subscription(
            VehicleRatesSetpoint, drone_prefix+'/fmu/out/vehicle_rates_setpoint', self.vehicle_rates_setpoint_callback, qos_drone)
        self.trajectory_setpoint_out_subscriber = self.create_subscription(
            TrajectorySetpoint, drone_prefix+'/fmu/out/trajectory_setpoint', self.trajectory_setpoint_out_callback, qos_drone)
        self.trajectory_setpoint_in_subscriber = self.create_subscription(
            TrajectorySetpoint, drone_prefix+'/fmu/in/trajectory_setpoint',self.trajectory_setpoint_in_callback, qos_drone)
        
        # Create publishers
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand,drone_prefix+'/fmu/in/vehicle_command', qos_drone)
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode,drone_prefix+'/fmu/in/offboard_control_mode', qos_drone)
        self.vehicle_rates_setpoint_publisher = self.create_publisher(VehicleRatesSetpoint, drone_prefix+'/fmu/in/vehicle_rates_setpoint', qos_drone)
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, drone_prefix+'/fmu/in/trajectory_setpoint', qos_drone)
        self.vehicle_attitude_setpoint_publisher = self.create_publisher(VehicleAttitudeSetpoint, drone_prefix+'/fmu/in/vehicle_attitude_setpoint', qos_drone)

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
        self.thrust_body = float(-0.41)
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
        self.vo_est_ned = VehicleLocalPosition()                # current state vector (estimated, NED)
        self.vo_ext = VehicleOdometry()                         # current state vector (external)
        self.vrs = VehicleRatesSetpoint()                       # current vehicle rates setpoint
        self.tsp_out = TrajectorySetpoint()                     # current trajectory setpoint
        self.tsp_in = TrajectorySetpoint()                      # current trajectory setpoint

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
    
    def internal_estimator_ned_callback(self, vehicle_local_position:VehicleLocalPosition):
        """Callback function for internal vehicle_local_position topic subscriber."""
        self.vo_est_ned = vehicle_local_position

    def external_estimator_callback(self, vehicle_odometry:VehicleOdometry):
        """Callback function for external vehicle_odometry topic subscriber."""
        self.vo_ext = vehicle_odometry

    def vehicle_rates_setpoint_callback(self, vehicle_rates_setpoint:VehicleRatesSetpoint):
        """Callback function for vehicle_rates topic subscriber."""
        self.vrs = vehicle_rates_setpoint

    def trajectory_setpoint_out_callback(self, trajectory_setpoint:TrajectorySetpoint):
        """Callback function for trajectory_setpoint topic subscriber."""
        self.tsp_out = trajectory_setpoint

    def trajectory_setpoint_in_callback(self, trajectory_setpoint:TrajectorySetpoint):
        """Callback function for trajectory_setpoint topic subscriber."""
        self.tsp_in = trajectory_setpoint

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

#FIXME
    def yaw_error_comp(self, similarity: np.ndarray):
        """
        Compute normalized horizontal yaw error (~[-1,1]) from a similarity/logit map.
        Positive => target left of center (command CCW yawspeed); negative => right (CW).
        """
        yaw_err_norm = 0.0
        try:
            S = similarity  # avoid an unnecessary copy
            if S is None:
                return 0.0

            S = S.astype(np.float32, copy=False)
            # Ignore negatives if logits: treat them as 0 weight
            np.maximum(S, 0.0, out=S)

            M = cv2.moments(S, binaryImage=False)  # intensity-weighted centroid
            m00 = M.get("m00", 0.0)
            if m00 > 1e-6:
                cx = M["m10"] / m00
                W = int(S.shape[1])
                cx_img = 0.5 * W
                # object right of center -> negative error -> CW yaw
                yaw_err_norm = float((cx - cx_img) / max(1.0, cx_img))
                # sanitize
                if not np.isfinite(yaw_err_norm):
                    yaw_err_norm = 0.0
                else:
                    yaw_err_norm = float(np.clip(yaw_err_norm, -1.0, 1.0))
            # else: leave at 0.0
            return yaw_err_norm
        except Exception:
            # Fail safe: neutral command
            return 0.0
        
    # def yaw_error_comp(self, similarity: np.ndarray):
    #     yaw_err_norm = 0.0
    #     cx = None
    #     S = similarity.copy()
    #     if S is not None:
    #         S = S.astype(np.float32)
    #         S = np.maximum(S, 0.0)
    #         M  = cv2.moments(S, binaryImage=False)
    #         if M["m00"] > 1e-6:
    #             cx = M["m10"] / M["m00"]
    #             W  = S.shape[1]
    #             cx_img = 0.5 * W
    #             yaw_err_norm = float((cx - cx_img) / max(1.0, cx_img))
    #     return yaw_err_norm

    def pid_yaw_rate(self, err_norm: float):
        Kp = getattr(self, "yaw_kp", 1.2)       # rad/s per unit error
        Ki = getattr(self, "yaw_ki", 0.0)       # keep 0 unless you really need it
        Kd = getattr(self, "yaw_kd", 0.5)      # rad/s per unit error derivative
      
        rate_max   = getattr(self, "yaw_rate_max", 0.8)  # clamp for safety
        i_max      = getattr(self, "yaw_i_limit", 0.5)   # integral windup guard
    
        # --- Stateful terms (lazy init) ---
        if not hasattr(self, "_yaw_i"):   self._yaw_i = 0.0
        if not hasattr(self, "_yaw_e0"):  self._yaw_e0 = 0.0
    
        # --- PID ---
        de = (err_norm - self._yaw_e0) / max(1e-3, self.dt)
        self._yaw_i = float(np.clip(self._yaw_i + err_norm*self.dt, -i_max, i_max))
        self._yaw_e0 = err_norm
    
        u = Kp*err_norm + Ki*self._yaw_i + Kd*de
        return float(np.clip(u, -rate_max, rate_max))
    
    def wrap_to_pi(self, a: float) -> float:
        # returns angle in (-pi, pi]
        return (a + np.pi) % (2*np.pi) - np.pi

    def rotate_vo_to_lpos(self, vx_vo: float, vy_vo: float, yaw_vo: float, heading_lpos: float) -> tuple[float, float]:
        """
        Rotate XY velocity from the VehicleOdometry yaw frame into VehicleLocalPosition yaw frame.
        yaw_vo: yaw from your VO/odometry pipeline (rad)
        heading_lpos: VehicleLocalPosition.heading (rad, NED)
        """
        dpsi = self.wrap_to_pi(heading_lpos - yaw_vo)
        c, s = float(np.cos(dpsi)), float(np.sin(dpsi))
        vx_lp =  c*vx_vo + s*vy_vo
        vy_lp = -s*vx_vo + c*vy_vo
        return float(vx_lp), float(vy_lp)
#
#NOTE COMMANDER
    def commander(self) -> None:
        """Main control loop for generating commands."""
        # Looping Callback Actions
        x_est,u_pr = zch.vo2x(self.vo_est,self.T_e2w)[0:10],zch.vrs2uvr(self.vrs)
        x_ext,obj = zch.vo2x(self.vo_ext)[0:10],self.obj
        x_est[6:10] = th.obedient_quaternion(x_est[6:10],self.xref[6:10])
        x_ext[6:10] = th.obedient_quaternion(x_ext[6:10],self.xref[6:10])

        frame = self.latest_frame
        # img = self.latest_mask
        similarity = self.latest_similarity
        # depth = self.latest_depth_xyz
        loiter_overlay = self.latest_loiter_overlay
#NOTE HEARTBEATS
        # zch.heartbeat_offboard_control_mode(self.get_current_timestamp_time(),self.offboard_control_mode_publisher)
        if self.sm == StateMachine.ACTIVE:
            # Active/Spin: use body-rate control
            if self.pilot_name == 'range_velocity':
                zch.heartbeat_offboard_control_mode(
                    self.get_current_timestamp_time(),
                    self.offboard_control_mode_publisher,
                    body_rate=False, position=True, velocity=True, attitude=False
                )
            else:
                zch.heartbeat_offboard_control_mode(
                    self.get_current_timestamp_time(),
                    self.offboard_control_mode_publisher,
                    body_rate=True, position=False, velocity=False
                )
        elif self.sm == StateMachine.HOLD:
            # Hold: use velocity control for smooth hover
            zch.heartbeat_offboard_control_mode(
                self.get_current_timestamp_time(),
                self.offboard_control_mode_publisher,
                body_rate=False,
                position=True,
                velocity=True
            )
        elif self.sm == StateMachine.SPIN:
            # for yaw-in-place: hold position via zero velocity, spin via body_rate
            zch.heartbeat_offboard_control_mode(
                self.get_current_timestamp_time(), self.offboard_control_mode_publisher,
                body_rate=False,  # drive the yaw-rate loop
                position=True,
                velocity=True   # drive the XY/Z velocity loops
            )
        else:
            # INIT, READY, LAND, etc.—default to body-rate so commands don’t break
            zch.heartbeat_offboard_control_mode(
                self.get_current_timestamp_time(),
                self.offboard_control_mode_publisher,
                body_rate=True,
                position=False,
                velocity=False
            )

    #FIXME        
        if self.key_pressed == '\x1b':     # ESC
            print("ESC detected, landing…")
            self.sm = StateMachine.LAND
            self.key_pressed = None        # reset
        elif self.key_pressed == 'h':      # H key
            print("H key detected, switching to HOLD mode…")
            self.alt_hold = float(x_est[2])
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
            self.alt_hold = float(x_est[2])
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
        elif self.key_pressed == 'p':      #
            print("P key detected, switching to TEST mode")
            self.alt_hold = float(x_est[2])
            self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
            zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            self.sm = StateMachine.TEST
            self.key_pressed = None        # reset
    #
###########################################
#NOTE              INIT
###########################################
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
###########################################
#NOTE                READY
###########################################
        elif self.sm == StateMachine.READY:
            # Check if we are ready to start (matching attitude and altitude)
            if np.linalg.norm(x_est[6:10]-self.q0) < self.q0_tol and np.abs(x_est[2]-self.z0) < self.z0_tol:
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
                self.k_rdy = 0                                                  # Reset ready counter
                zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
                self.sm = StateMachine.TEST
                
                # Print State Information
                print('=====================================================================')
                print('TEST Started.')
            else:
                # Looping State Actions
                self.k_rdy += 1
                if self.k_rdy % 10 == 0:
                    z0ds,z0cr = np.around(self.z0,2),np.around(x_est[2],2)
                    q0ds,q0cr = np.around(self.q0,2),np.around(x_est[6:10],2)

                    print(f"Desired/Current Altitude+Attitude: ({z0ds}/{z0cr})({q0ds}/{q0cr})")
###########################################
#NOTE                TEST
###########################################
        elif self.sm == StateMachine.TEST:
            x_cur = x_est
            t0_lp  = time.time()                                                # Algorithm start time
            t_tr = self.get_current_trajectory_time()

            zch.publish_velocity_and_altitude_hold(
                    self.get_current_timestamp_time(),
                    self.alt_hold,
                    self.trajectory_setpoint_publisher
                )
            
            if t_tr < (5.0):
                # Determine yaw rate using PID
                k_gate       = 1.5
                yaw_err_norm = self.yaw_error_comp(similarity)
                yaw_rate_cmd = self.pid_yaw_rate(yaw_err_norm)
                yaw_err      = float(np.clip(yaw_err_norm, -1.0, 1.0))
                alpha        = float(np.clip(1.0 - k_gate * abs(yaw_err), 0.0, 1.0))

                R_b2w          = zch.quat_xyzw_to_R(*x_cur[6:10].astype(np.float64))
                fwd            = R_b2w[:,0]
                # vx_dir, vy_dir = fwd[0], fwd[1]
                # norm_xy        = np.hypot(vx_dir, vy_dir)

                # if norm_xy > 1e-6:
                #     vx_dir, vy_dir = vx_dir/norm_xy, vy_dir/norm_xy

                # v_fwd          = float(np.clip(1.0 * alpha, 0.0, 1.0))
                # vx, vy         = v_fwd * vx_dir, v_fwd * vy_dir

                yaw_world      = float(np.arctan2(R_b2w[1, 0], R_b2w[0, 0]))
                v_fwd          = float(np.clip(1.0 * alpha, 0.0, 1.0))
                vx, vy         = v_fwd * np.cos(yaw_world), v_fwd * np.sin(yaw_world)

                # Px4 does something insane with velocities here, so we have to transform
                v_w       = np.array([vx, vy, 0.0], dtype=float)
                R_w2b     = R_b2w.T
                v_b_frd   = R_w2b @ v_w                                  # body FRD (X fwd, Y right, Z down)
                v_b_flu   = np.array([v_b_frd[0], -v_b_frd[1], -v_b_frd[2]], dtype=float)  # body FLU

                vx_send, vy_send = vx, vy

################DEBUG#####################
                now_s = time.time()
                if not hasattr(self, "_last_active_print"):
                    self._last_active_print = 0.0
                if now_s - self._last_active_print > 1.0:   # 1 Hz
                    self._last_active_print = now_s

                    pos = x_cur[0:3]
                    vel = x_cur[3:6]

                    yaw_from_fwd = np.degrees(np.arctan2(fwd[1], fwd[0]))

                    # qx, qy, qz, qw = x_cur[6:10]
                    # rpy = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)

                    print("=== TEST DEBUG ===")

                    print(f"Pos (m): {pos}")
                    print(f"{('Vel (m/s): EST. ' + str(vel.tolist())):<48} "
                          f"CMP. [{vx:.3f}, {vy:.3f}, 0.000] "
                          f"CMD. [{vx_send:.3f}, {vy_send:.3f}, 0.000] "
                          f"PUB-IN. [{self.tsp_in.velocity[0]:.3f}, {self.tsp_in.velocity[1]:.3f}, {self.tsp_in.velocity[2]:.3f}] "
                          f"PUB-OUT. [{self.tsp_out.velocity[0]:.3f}, {self.tsp_out.velocity[1]:.3f}, {self.tsp_out.velocity[2]:.3f}] ")
                    print(f"Yaw (planar) deg: {np.degrees(yaw_world):.2f} (from fwd: {yaw_from_fwd:.2f})")
                    print(f"Commanded yawspeed (rad/s): {yaw_rate_cmd:.3f}")
                    print("====================")
###########################################
                t_sol = np.hstack((0.0,0.0,0.0,0.0,time.time()-t0_lp))
                u_cmd = np.array([vx, vy, 0.0, yaw_rate_cmd], dtype=float)  # shape (nu,)

                # Minimal refs/aux (placeholders; sizes must match shapes the recorder was built with)
                xref = np.zeros(self.recorder.Xref.shape[0], dtype=float)   # shape (nx,)
                xref[2] = self.alt_hold
                uref = np.zeros(self.recorder.Uref.shape[0], dtype=float)   # shape (nu,)
                adv  = np.zeros(self.recorder.Adv.shape[0],  dtype=float)   # shape (nu,)
                # tsol = np.zeros(self.recorder.Tsol.shape[0], dtype=float)   # Ntsol

                # Record (image can be your latest frame or a blank stub of the right size)
                # img = self.latest_frame if self.latest_frame is not None else np.zeros_like(self.recorder.Imgs[0])
                S = np.maximum(similarity, 0).astype(np.float32)                        # clip negatives if logits
                S8 = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                S_bgr = cv2.cvtColor(S8, cv2.COLOR_GRAY2BGR) 
                # self.recorder.record(S_bgr)
                self.recorder.record(
                    img=S_bgr,
                    tcr=t_tr,
                    ucr=u_cmd,
                    xref=xref,
                    uref=uref,
                    xest=x_est,
                    xext=x_est,
                    adv=adv,
                    tsol=t_sol
                )
            else:
                # trajectory has ended → HOLD Position
                self.alt_hold = float(x_est[2])
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9                       # Record start time
                self.policy_duration       = t_tr
                self.hold_prompt           = self.prompt

                self.spin_cycle = False

                self.ready_active = False                                                 # Reset ready active flag
                self.found = False
                self.sm                    = StateMachine.HOLD
                print('Trajectory Finished → HOLDing Position, readying for next query...')
###########################################
#NOTE                ACTIVE
###########################################
        elif self.sm == StateMachine.ACTIVE:
            # Looping State Actions
            x_cur = x_est
            t0_lp  = time.time()                                                # Algorithm start time
            t_tr = self.get_current_trajectory_time()                           # Current trajectory time

            sem_exit, reason, m = self.vision_model._query_found(similarity)
            
            # Check if we are still in the trajectory
            if t_tr < (self.Tpi[-1]+self.t_lg) and not sem_exit:
                # keep your references if you record them
                t_ref = np.min((t_tr, self.Tpi[-1]))
                x_ref = self.xref
                u_ref = self.uref
            #FIXME
                # --- vo_ext readout: position, velocity, orientation (planar yaw) ---
                pos_ext = x_ext[0:3]
                vel_ext = x_ext[3:6]
                R_b2w_ext = zch.quat_xyzw_to_R(*x_ext[6:10].astype(np.float64))
                yaw_ext = float(np.arctan2(R_b2w_ext[1, 0], R_b2w_ext[0, 0]))
                yaw_ext_deg = float(np.degrees(np.arctan2(R_b2w_ext[1, 0], R_b2w_ext[0, 0])))
######################################INTRACTIBLE#######################################
                # Determine yaw rate using PID
                k_gate       = 1.5
                yaw_err_norm = self.yaw_error_comp(similarity)
                yaw_rate_cmd = self.pid_yaw_rate(yaw_err_norm)
                yaw_err      = float(np.clip(yaw_err_norm, -1.0, 1.0))
                alpha        = float(np.clip(1.0 - k_gate * abs(yaw_err), 0.0, 1.0))

                R_b2w          = zch.quat_xyzw_to_R(*x_cur[6:10].astype(np.float64))
                fwd            = R_b2w[:,0]
                # vx_dir, vy_dir = fwd[0], fwd[1]
                # norm_xy        = np.hypot(vx_dir, vy_dir)

                # if norm_xy > 1e-6:
                #     vx_dir, vy_dir = vx_dir/norm_xy, vy_dir/norm_xy

                # v_fwd          = float(np.clip(1.0 * alpha, 0.0, 1.0))
                # vx, vy         = v_fwd * vx_dir, v_fwd * vy_dir

                yaw_world      = float(np.arctan2(R_b2w[1, 0], R_b2w[0, 0]))
                v_fwd          = float(np.clip(1.0 * alpha, 0.0, 1.0))
                vx, vy         = v_fwd * np.cos(yaw_world), v_fwd * np.sin(yaw_world)
                # vx, vy         = v_fwd * np.cos(yaw_ext), v_fwd * np.sin(yaw_ext)

                # Px4 does something insane with velocities here, so we have to transform
                # v_w       = np.array([vx, vy, 0.0], dtype=float)
                # R_w2b     = R_b2w.T
                # v_b_frd   = R_w2b @ v_w                                  # body FRD (X fwd, Y right, Z down)
                # v_b_flu   = np.array([v_b_frd[0], -v_b_frd[1], -v_b_frd[2]], dtype=float)  # body FLU

                vx_send, vy_send = self.rotate_vo_to_lpos(vx, vy, yaw_world, self.vo_est_ned.heading)

                # vx_send, vy_send = vx, vy
                # vx_send, vy_send = v_b_flu[0], v_b_flu[1]
                # vx_send = -1.0*v_fwd
                # vy_send = 0.0

################DEBUG#####################
                now_s = time.time()
                if not hasattr(self, "_last_active_print"):
                    self._last_active_print = 0.0
                if now_s - self._last_active_print > 1.0:   # 0.5 Hz
                    self._last_active_print = now_s

                    pos = x_cur[0:3]
                    vel = x_cur[3:6]

                    yaw_from_fwd = np.degrees(np.arctan2(fwd[1], fwd[0]))

                    print("=== ACTIVE DEBUG ===")
                    # print(f"Pos (m): EST-INT {pos}")
                    print(f"{('Pos (m): EST' + np.array2string(pos, precision=3, separator=', ')):<48} "
                          f"EXT. [{vel_ext[0]:.3f}, {vel_ext[1]:.3f}, {vel_ext[2]:.3f}] ")
                    print(f"{('Vel (m/s): EST. ' + str(vel.tolist())):<48} "
                          f"CMP. [{vx:.3f}, {vy:.3f}, 0.000] "
                          f"CMD. [{vx_send:.3f}, {vy_send:.3f}, 0.000] "
                          f"PUB-IN. [{self.tsp_in.velocity[0]:.3f}, {self.tsp_in.velocity[1]:.3f}, {self.tsp_in.velocity[2]:.3f}] "
                          f"PUB-OUT. [{self.tsp_out.velocity[0]:.3f}, {self.tsp_out.velocity[1]:.3f}, {self.tsp_out.velocity[2]:.3f}] ")
                    # print(f"{('Vel (m/s): EST. ' + str(vel.tolist())):<48} COMP. [{vx:.3f}, {vy:.3f}, 0.000] SENT. [{vx_send:.3f}, {vy_send:.3f}, 0.000]")
                    # print(f"Yaw (planar) deg: {np.degrees(yaw_world):.2f}")
                    print(f"Yaw (planar) deg: {np.degrees(yaw_world):.2f} (from fwd: {yaw_from_fwd:.2f}) EXT: {yaw_ext_deg:.2f}")
                    print(f"Commanded yawspeed (rad/s): {yaw_rate_cmd:.3f}")

                    print("====================")
###########################################

                # zch.publish_visual_servoing_setpoint(
                #     self.get_current_timestamp_time(),
                #     self.thrust_body,
                #     float('nan'),
                #     float('nan'),
                #     yaw_rate_cmd,
                #     self.vehicle_attitude_setpoint_publisher)
                zch.publish_visual_servoing_setpoint(
                    self.get_current_timestamp_time(),
                    self.alt_hold,
                    yaw_rate_cmd,
                    np.array([vx_send, vy_send, 0.0], dtype=np.float32),
                    self.trajectory_setpoint_publisher)
                
                t_sol = np.hstack((0.0,0.0,0.0,0.0,time.time()-t0_lp))
                # u_cmd = np.array([self.thrust_body, 0.0, 0.0, yaw_rate_cmd], dtype=float)  # shape (nu,)
                u_cmd = np.array([vx_send, vy_send, 0.0, yaw_rate_cmd], dtype=float)  # shape (nu,)

                # Minimal refs/aux (placeholders; sizes must match shapes the recorder was built with)
                xref = np.zeros(self.recorder.Xref.shape[0], dtype=float)   # shape (nx,)
                xref[2] = self.alt_hold
                uref = np.zeros(self.recorder.Uref.shape[0], dtype=float)   # shape (nu,)
                adv  = np.zeros(self.recorder.Adv.shape[0],  dtype=float)   # shape (nu,)
                # tsol = np.zeros(self.recorder.Tsol.shape[0], dtype=float)   # Ntsol

                # Record (image can be your latest frame or a blank stub of the right size)
                # img = self.latest_frame if self.latest_frame is not None else np.zeros_like(self.recorder.Imgs[0])
                S = np.maximum(similarity, 0).astype(np.float32)                        # clip negatives if logits
                S8 = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                S_bgr = cv2.cvtColor(S8, cv2.COLOR_GRAY2BGR) 
                # self.recorder.record(S_bgr)
                self.recorder.record(
                    img=S_bgr,
                    tcr=t_tr,
                    ucr=u_cmd,
                    xref=xref,
                    uref=uref,
                    xest=x_est,
                    xext=x_est,
                    adv=adv,
                    tsol=t_sol
                )

                # # --- forward velocity in *world frame* along current yaw ---
                # # NOTE: PX4 TrajectorySetpoint.velocity is in local frame; convert "forward"
                # # (body-x) to local XY using the current yaw.
                # qx, qy, qz, qw = x_est[6:10]
                # yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz')[2]
            
                # v_fwd = getattr(self, "forward_speed_mps", 1.0)   # gentle ~0.1 m/s
                # vx = float(v_fwd * np.cos(yaw))
                # vy = float(v_fwd * np.sin(yaw))
                # vz = 0.0
            
                # # Publish a velocity-only TrajectorySetpoint with yawspeed
                # ts = TrajectorySetpoint(timestamp=int(self.get_current_timestamp_time()))
                # ts.position     = [float('nan'), float('nan'), float(self.alt_hold)]#[float('nan')]*3
                # ts.velocity     = [vx, vy, float('nan')]
                # ts.acceleration = [float('nan')]*3
                # ts.jerk         = [float('nan')]*3
                # ts.yaw          = float('nan')           # NaN → use yawspeed
                # ts.yawspeed     = float(yaw_rate_cmd)
                # self.trajectory_setpoint_publisher.publish(ts)
            #
            else:
                # trajectory has ended → HOLD Position
                self.alt_hold = float(x_est[2])
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

###########################################
#NOTE                HOLD
###########################################
        elif self.sm == StateMachine.HOLD:
            t_tr = self.get_current_trajectory_time()                           # Current trajectory time
            # zch.publish_velocity_hold(
            #     self.get_current_timestamp_time(),
            #     self.trajectory_setpoint_publisher
            # )
        #NOTE THIS SHOULD BE REPLACED WITH A NEWER VERSION OF THE READY STATE, BUT FOR NOW IT SUFFICES
            if (self.ready_active is True and self.spin_cycle is False) and t_tr > 3.0:
                print("Detector Thread Sleeping; Switching to Active")
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
                self.ready_active = False                                       # Reset ready active flag
                self.found = False
                zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
                self.sm = StateMachine.ACTIVE
                
                print('=====================================================================')
                print('ACTIVE Started.')
            elif (self.ready_active is False and self.spin_cycle is True) and t_tr > 3.0:
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
                self.alt_hold = float(x_est[2])
                zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
                self.sm = StateMachine.SPIN
                
                print('=====================================================================')
                print('SPIN Started.')
            elif (self.ready_active is True and self.spin_cycle is False) and t_tr < 3.0:
                yaw_err_norm = self.yaw_error_comp(similarity)
            
                # --- PID → yaw-rate and hold velocities while we re-center the mask ---
                yaw_rate_cmd = self.pid_yaw_rate(yaw_err_norm)
            
                # Use existing helper: zero-velocity hold + commanded yaw-rate (no new APIs)
                zch.publish_velocity_and_altitude_hold_with_yaw_rate(
                    self.get_current_timestamp_time(),
                    self.alt_hold,
                    yaw_rate_cmd,
                    self.trajectory_setpoint_publisher
                )
            elif t_tr < 6.0:
                zch.publish_velocity_and_altitude_hold(
                    self.get_current_timestamp_time(),
                    self.alt_hold,
                    self.trajectory_setpoint_publisher
                )
            else:
                self.sm = StateMachine.LAND
                print('Trajectory Finished → Landing...')
            self.recorder.record(frame)
###########################################
#NOTE                SPIN
###########################################
        elif self.sm == StateMachine.SPIN:
            t_tr = self.get_current_trajectory_time()

            zch.publish_velocity_and_altitude_hold_with_yaw_rate(
                self.get_current_timestamp_time(),
                self.alt_hold,
                0.9,
                self.trajectory_setpoint_publisher
                )
            if t_tr < 8.0:
                self.active_arm = False
                # self.recorder.record(img)
            elif t_tr >= 8.0 and t_tr < 24.0:
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
                    
                    self.ready_active = True
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
###########################################
#NOTE               LAND
###########################################
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

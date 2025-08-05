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
        self._cleanup_done = False
        # ---------------------------------
        # Register cleanup routines
        sys.excepthook = self._handle_exception
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)
        atexit.register(self._cleanup)
        # ---------------------------------
#FIXME
        # --- curses setup ---
        # self.stdscr = curses.initscr()
        # curses.noecho()
        # curses.cbreak()
        # self.stdscr.keypad(True)

        # h, w = self.stdscr.getmaxyx()
        # # status window (all your print()s go here)
        # self.status_win = curses.newwin(h - 3, w, 0, 0)
        # self.status_win.scrollok(True)

        # self._orig_stdout = sys.stdout
        # self._orig_stderr = sys.stderr
        # sys.stdout = sys.stderr = self

        # # input window (bottom 3 lines)
        # self.input_win = curses.newwin(3, w, h - 3, 0)
        # self.input_win.border()
        # # make getch() non-blocking on this window
        # self.input_win.nodelay(True)
        # self.input_win.refresh()

        # # monkey-patch print → status_win
        # self._orig_print = builtins.print
        # builtins.print = self.curses_print

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

#FIXME
        print("Loading Configs")
        # Load Configs
        workspace_path = os.path.dirname(os.path.dirname(__file__))

        # Check for mission file
        if os.path.isfile(os.path.join(workspace_path,"configs","missions",mission_name+".json")):
            with open(os.path.join(workspace_path,"configs","missions",mission_name+".json")) as json_file:
                mission_config = json.load(json_file)
        else:
            raise FileNotFoundError(f"Mission file not found: {mission_name}.json")
#FIXME
        print("Loading Courses")
        # with open(os.path.join(workspace_path,"configs","courses",mission_config["course"]+".json")) as json_file:
            # mission_config = json.load(json_file)
#NOTE modified this approach to load the RRT* generated tXUi from the pickle file
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
#FIXME
        print("Loaded Frame")
        #NOTE RRT*/SSV precomputes the trajectory, we load it from a saved file
        # # Unpack trajectories
        # output = ms.solve(course_config)
        # if output is not False:
        #     Tpi, CPi = output
        # else:
        #     raise ValueError("Trajectory not feasible. Aborting.")
        # tXUi = th.ts_to_tXU(Tpi,CPi,None,hz)
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
#FIXME:
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
        else:
            self.isNN = True
            self.ctl = Pilot(mission_config["cohort"],mission_config["pilot"])
            self.ctl.set_mode('deploy')

        # Create subscribers
        self.internal_estimator_subscriber = self.create_subscription(
            VehicleOdometry, drone_prefix+'/fmu/out/vehicle_odometry', self.internal_estimator_callback, qos_drone)
        self.external_estimator_subscriber = self.create_subscription(
            VehicleOdometry, '/drone0/fmu/in/vehicle_visual_odometry', self.external_estimator_callback, qos_mocap)
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
        self.spin_ready            = False
        self.spin_thread_started   = False
        self.prompt_thread_started = False
        self.prompt_buffer         = None
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
        self.pipeline = zch.get_camera(cam_dim[0],cam_dim[1],cam_fps)
        if self.pipeline is not None:
            self.cam_dim,self.cam_fps = cam_dim,cam_fps
        else:
            self.cam_dim,self.cam_fps = [0,0,0],0
        self.img_times = []

        self.vision_thread     = None
        self.vision_shutdown   = False
        self.vision_started    = False
        self.latest_mask       = None

#NOTE streamline testing non-mpc pilots
        if not self.isNN:
            # MPC case: unpack the precomputed trajectory exactly as you do today
            self.Tpi = tXUi[0, :]                             # time vector
            self.obj = th.tXU_to_obj(tXUi)                    # your “object” vector
            self.q0, self.z0 = tXUi[7:11,0], tXUi[3,0]         # initial quaternion & altitude
            self.xref = tXUi[1:11,0]                          # 10-state at t=0
            mean_u = np.mean(tXUi[14:18,:], axis=0).reshape(1,-1)
            self.uref = np.vstack((-mean_u, tXUi[11:14,:]))    # reference inputs over time

            # seed your odometry messages to match the first pose in the traj
            self.vo_est.q[0], self.vo_est.q[1:4] = tXUi[10,0], tXUi[7:10,0]
            self.vo_ext.q[0], self.vo_ext.q[1:4] = tXUi[10,0], tXUi[7:10,0]

            # record for the full traj duration + linger
            record_duration = self.Tpi[-1] + self.t_lg

        else:
            # non-MPC (neural) case: give everything a neutral default
            dt = 1.0 / hz          # hz (control rate)
            self.Tpi = np.arange(0.0, 1.0 + dt, dt)
            self.obj = np.zeros((18,1))
            self.q0 = np.array([0.0, 0.0, 0.70710678, 0.70710678])          # 
            self.z0 = -0.50                                    # init altitude offset
            self.xref = np.zeros(10)                          # no reference state
            self.uref = -6.90*np.ones((4,))                   # reference input
            # leave vo_est/.vo_ext.q alone so they start at whatever your first callback gives
            record_duration = self.Tpi[-1] + self.t_lg        # 


        # Initialize control variables
#NOTE commented out variables don't exist for RRT* generated tXUi
        self.node_start = False                                 # Node start flag
        self.node_status_informed = False                       # Node status informed flag
        # self.Tpi,self.CPi = Tpi,CPi                             # trajectory time and control points

        self.drone_config = drone_config                        # drone configuration
        # self.k_sm,self.N_sm = 0, len(Tpi)-1                     # current segment index and total number of segments
        self.k_rdy = 0                                          # ready counter
        self.t_tr0 = 0.0                                        # trajectory start time
        self.vo_est = VehicleOdometry()                         # current state vector (estimated)
        self.vo_ext = VehicleOdometry()                         # current state vector (external)
        self.vrs = VehicleRatesSetpoint()                       # current vehicle rates setpoint
        self.znn = (torch.zeros(self.ctl.model.Nz)              # current visual feature vector
                    if isinstance(self.ctl,Pilot) else None)   
#FIXME
        # Initialize Quaternions with starting reference quaternion (note: tXUi convention is qx,qy,qz,qw while vo_est is qw,qx,qy,qz)
        # self.vo_est.q[0],self.vo_est.q[1:4] = tXUi[10,0],tXUi[7:10,0]
        # self.vo_ext.q[0],self.vo_ext.q[1:4] = tXUi[10,0],tXUi[7:10,0]

        # State Machine and Failsafe variables
        self.sm = StateMachine.INIT                                    # state machine
        # self.xref = tXUi[1:11,0]                                       # reference state
#NOTE referring to something used in main loop here
        # u_ref = np.hstack((-np.mean(xu_ref[13:17]),xu_ref[10:13]))
        # print(f"mean of tXUi[14:18,0]: {np.mean(tXUi[14:18,:],axis=0)}")
        # mean_tXUi = np.mean(tXUi[14:18, :], axis=0)
        # mean_tXUi = mean_tXUi.reshape(1, -1)
        # print(f"mean_tXUi: {mean_tXUi.shape}")
        # print(f"tXUi[11:14,0]: {tXUi[11:14,:].shape}")
        # self.uref = np.vstack((-mean_tXUi,tXUi[11:14,:])) # reference input
        
        # self.q0,self.z0 = tXUi[7:11,0],tXUi[3,0]                       # initial attitude and altitude
        self.q0_tol,self.z0_tol = q0_tol,z0_tol                        # attitude and altitude tolerances

        # Create Variables for logging.
        self.recorder = rf.FlightRecorder(
            drone_config["nx"],
            drone_config["nu"],
            self.ctl.hz,
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
        self.cmdLoop = self.create_timer(1/self.ctl.hz, self.commander)

        # Earth to World Pose
        self.T_e2w = np.eye(4)

        # Diagnostics Variables
#NOTE Maybe save the nodes from RRT* here? could be useful for diagnostics
        # table = []
        # for idx,item in enumerate(course_config["keyframes"].values()):
        #     FOkf = np.array(item['fo'],dtype=float)
        #     data = np.hstack((self.Tpi[idx],FOkf[:,0]))
        #     table.append(data)

        # Print Diagnostics
        print('=====================================================================')
        print("========================= SFTI COMMAND NODE =========================")
        print("---------------------------------------------------------------------")
        print('-----------------------> Node Configuration <------------------------')
        print('Cohort Name      :',mission_config["cohort"])
        print('Scene Name      :',mission_config["scene"])
        print('Pilot Name       :',mission_config["pilot"])
        print('Quad Name        :',drone_config["name"])
        print('Control Frequency:',self.ctl.hz)
        print('Images Per Second:',mission_config["images_per_second"])
        print('Failsafe Tolerances (q_init,z_init):',mission_config["failsafes"]["attitude_tolerance"],mission_config["failsafes"]["height_tolerance"])
        print('=====================================================================')
        if self.isNN is False:
            Qk,Rk = np.diagonal(self.ctl.Qk),np.diagonal(self.ctl.Rk)
            QN = np.diagonal(self.ctl.QN)

            print('--------------------------> MPC Weights <----------------------------')
            print('Stagewise Weights:')
            print('Position:',Qk[0:3])
            print('Velocity:',Qk[3:6])
            print('Attitude:',Qk[6:10])
            print('')
            print('Thrust  :',Rk[0])
            print('Rates   :',Rk[1:4])
            print('')
            print('Terminal Weights:')
            print('Position:',QN[0:3])
            print('Velocity:',QN[3:6])
            print('Attitude:',QN[6:10])
            print('=====================================================================')

        print('--------------------> Trajectory Information <-----------------------')
        # print('Number of Segments :',self.N_sm)
        # print('Trajectory Duration:',tXUi[0,-1],'s')
        print('---------------------------------------------------------------------')
        # print('Keyframes:')
        print('Starting Location & Orientation:')
        x0_est = zch.vo2x(self.vo_est,self.T_e2w)[0:10]
        x0_est[6:10] = th.obedient_quaternion(x0_est[6:10],self.xref[6:10])
        print('xref: ',x0_est[1:3])
        print('qref: ',x0_est[6:10])
        # print('xref',self.xref)
        # print(tabulate(table, headers=["x (m)", "y (m)", "z (m)", "yaw (rad)"]))
        print('---------------------------------------------------------------------')
        print('=====================================================================')

        # Plot Trajectory
        # ps.CP_to_3D([self.Tpi],[self.CPi],n=50)

        # Wait for GCS clearance
#FIXME
        # self.input_win.clear()
        # self.input_win.border()
        # self.input_win.addstr(1, 2, "Press Enter to proceed…")
        # self.input_win.refresh()

        # # turn blocking on just for this prompt
        # self.input_win.nodelay(False)
        # while True:
        #     ch = self.input_win.getch()
        #     if ch in (10, 13):   # Enter
        #         break
        # # go back to non-blocking for the rest of the run
        # self.input_win.nodelay(True)
        # self.input_win.clear()
        # self.input_win.border()
        # self.input_win.refresh()
#
        input("Press Enter to proceed...")

#FIXME
    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        # Called on any uncaught exception
        self._cleanup()
        # then show the normal Python traceback
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def _handle_signal(self, signum, frame):
        # Called on Ctrl-C or kill
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        # only run once
        if self._cleanup_done:
            return
        self._cleanup_done = True

        # 2) Stop & join threads
        if getattr(self, 'vision_thread', None) and self.vision_started:
            print("Closing Vision Thread")
            self.vision_shutdown = True
            self.vision_thread.join(timeout=1.0)
            self.vision_started = False
            print("Vision Thread Offline")

        if getattr(self, 'kb_thread', None):
            print("Closing Keyboard Thread")
            self.kb_shutdown.set()
            self.kb_thread.join(timeout=1.0)
            print("Keyboard Thread Offline")

        # 3) Camera, controller, recorder
        try:
            if self.pipeline:
                zch.close_camera(self.pipeline)
            if not self.isNN:
                self.ctl.clear_generated_code()
            _ = self.recorder.save()
        except Exception as e:
            print(f"Error during device cleanup: {e}")

        # 4) Tear down ROS node & context
        try:
            self.destroy_node()
        except Exception as e:
            print(f"Ignoring destroy_node error: {e}")
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"Ignoring rclpy.shutdown() error: {e}")
    # def _cleanup(self):
    #     try:
    #         # State Actions
    #         zch.land(self.get_current_timestamp_time(),
    #                  self.vehicle_command_publisher)

    #         if self.vision_thread is not None and self.vision_started:
    #             print("Closing Vision Thread")
    #             self.vision_shutdown = True
    #             self.vision_thread.join(timeout=1.0)
    #             self.vision_started = False
    #             print("Vision Thread Offline")

    #         # Close the camera (if exists)
    #         if self.pipeline is not None:
    #             zch.close_camera(self.pipeline)

    #         # Close the controller (if mpc)
    #         if not self.isNN:
    #             self.ctl.clear_generated_code()

    #         # Save data
    #         output_path = self.recorder.save()

    #     except Exception as e:
    #         # if cleanup itself blows up, we still want to know
    #         print(f"Error during cleanup: {e}")

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

    def async_vision_loop(self):
        """Continuously grab + infer, store only the newest mask."""
        while not self.vision_shutdown:
            imgz, _ = zch.get_image(self.pipeline)
            if imgz is None:
                time.sleep(0.01)
                continue
            t0_img = time.time()
            frame = cv2.cvtColor(imgz, cv2.COLOR_BGR2RGB)
            mask = self.vision_model.clipseg_hf_inference(
                frame,
                self.prompt,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False
            )
            self.latest_mask = mask
            t_elap = time.time() - t0_img
            self.img_times.append(t_elap)

    def quat_to_yaw(self,q):
        x, y, z, w = q
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def build_yaw_qp(self, xyz, yaw0, t0):
        """Background build of the spin trajectory."""
        cfg = th.generate_spin_keyframes(
            name=f"loiter_spin_{t0:.2f}",
            Nco=6,
            xyz=xyz,
            theta0=yaw0, theta1=yaw0,
            time=5.0
        )
        out = ms.solve(cfg)
        if out is False:
            print(f"Spin QP failed at t0={t0:.2f}")
            return
        self.Tps_spin, self.CPs_spin = out
        self.tXUd_spin       = th.TS_to_tXU(self.Tps_spin, self.CPs_spin,
                                            self.drone_config,
                                            self.ctl.hz)
        # self.spin_start_time = t0
        self.spin_ready     = True
        print("Spin QP ready")

#FIXME
    def curses_print(self, *args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        self.status_win.addstr(msg + "\n")
        self.status_win.refresh()

    def write_input(self):
        self.input_win.clear()
        self.input_win.border()
        # Always show prompt label
        label = "Query: "
        display = self.input_text
        max_x = self.input_win.getmaxyx()[1] - len(label) - 4
        if len(display) > max_x:
            display = display[-max_x:]
        self.input_win.addstr(1, 2, label + display)
        self.input_win.refresh()

    def write(self, msg):
        # this gets called for every .write() to stdout/stderr
        self.status_win.addstr(msg)
        self.status_win.refresh()

    def flush(self):
        # called for .flush(), can be a no-op
        pass

    def destroy_node(self):
        # once-only terminal restore + ROS‐publisher teardown
        if getattr(self, '_destroyed', False):
            return
        self._destroyed = True

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_tty)
        super().destroy_node()
#NOTE REDUNDANT
    # def destroy_node(self):
    #     # restore terminal so we don’t break your shell
    #     termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_tty)
    #     super().destroy_node()

    # def destroy_node(self):
    #     # restore terminal so we don’t break your shell
    #     # os.dup2(self._stdout_fd_dup, 1)
    #     # os.dup2(self._stderr_fd_dup, 2)
    #     # os.close(self._stdout_fd_dup)
    #     # os.close(self._stderr_fd_dup)

    #     # sys.stdout = self._orig_stdout
    #     # sys.stderr = self._orig_stderr
    #     # sys.stdout = self._orig_stdout
    #     # sys.stderr = self._orig_stderr
    #     # builtins.print = self._orig_print

    #     # curses.nocbreak()
    #     # curses.echo()
    #     # curses.endwin()
    #     if self._destroyed:
    #         return
    #     self._destroyed = True
    #     termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_tty)
    #     super().destroy_node()

    def wait_for_prompt(self):
        """Daemon thread: only prompt/get a line while in HOLD."""
        while True:
            # 1) spin-wait until we're in HOLD
            if self.sm != StateMachine.HOLD:
                time.sleep(0.1)
                continue

            # 2) now we’re in HOLD → actually prompt
            self.input_win.clear(); self.input_win.border()
            prompt = "Type your query and press Enter: "
            self.input_win.addstr(1, 2, prompt)
            self.input_win.refresh()
            curses.echo()
            s = self.input_win.getstr(1, len(prompt) + 2).decode()
            curses.noecho()

            # 3) stash it for the HOLD→SPIN transition
            self.prompt_buffer = s

            # 4) clear the box, then loop back (won’t re‐prompt until we re‐enter HOLD)
            self.input_win.clear(); self.input_win.border()
            self.input_win.refresh()

            # small pause so we don’t immediately re‐prompt on the same keypress
            time.sleep(0.1)
#
    def commander(self) -> None:
        """Main control loop for generating commands."""
        # Looping Callback Actions
        x_est,u_pr = zch.vo2x(self.vo_est,self.T_e2w)[0:10],zch.vrs2uvr(self.vrs)
        x_ext,znn,obj = zch.vo2x(self.vo_ext)[0:10],self.znn,self.obj
        x_est[6:10] = th.obedient_quaternion(x_est[6:10],self.xref[6:10])
        x_ext[6:10] = th.obedient_quaternion(x_ext[6:10],self.xref[6:10])

        img = self.latest_mask

        # zch.heartbeat_offboard_control_mode(self.get_current_timestamp_time(),self.offboard_control_mode_publisher)
        if self.sm == StateMachine.ACTIVE:
            # Active/Spin: use body-rate control
            zch.heartbeat_offboard_control_mode(
                self.get_current_timestamp_time(),
                self.offboard_control_mode_publisher,
                body_rate=True,
                velocity=False
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
        if self.key_pressed == '\x1b':    # ESC
            print("ESC detected, landing…")
            self.sm = StateMachine.LAND
            self.key_pressed = None        # reset
        elif self.key_pressed == 'h':    # H key
            print("H key detected, switching to HOLD mode…")
            self.hold_state = x_est.copy()  # save current state
            self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
            zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            self.sm = StateMachine.HOLD
            self.key_pressed = None        # reset
        elif self.key_pressed == 's':    # s key
            print("H key detected, switching to SPIN mode…")
            self.hold_state = x_est.copy()  # save current state
            self.t_tr0 = self.get_clock().now().nanoseconds/1e9             # Record start time
            zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            self.sm = StateMachine.SPIN
            self.key_pressed = None        # reset

        # ch = self.input_win.getch()
        # if ch != curses.ERR:
        #     if ch == 27:
        #         self.key_pressed = '\x1b'
        #     elif ch in (10, 13):
        #         self.key_pressed = '\n'
        #     else:
        #         # if you ever care about other single-char keys:
        #         try:
        #             self.key_pressed = chr(ch)
        #         except ValueError:
        #             self.key_pressed = None
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
                        target=self.async_vision_loop,
                        daemon=True,
                        name="ZedVisionThread"
                    )
                    self.vision_thread.start()
                    self.vision_started = True
                print("Vision Thread Online")

                # Print State Information
                print('---------------------------------------------------------------------')
                print('Node is ready. Starting trajectory.')
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
                # print('In Segment:',self.k_sm+1,'/',self.N_sm)
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

            # Check if we are still in the trajectory
            if t_tr < (self.Tpi[-1]+self.t_lg):
                # Compute the reference trajectory current point
                t_ref = np.min((t_tr,self.Tpi[-1]))                             # Reference time (does not exceed ideal)

                x_ref = self.xref
                u_ref = self.uref

                # Generate vrs command
                u_act,self.znn,adv,t_sol_ctl = self.ctl.control(t_tr,x_est,u_pr,obj,img,znn)

                # Send command
                zch.publish_uvr_command(self.get_current_timestamp_time(),u_act,self.vehicle_rates_setpoint_publisher)
                
                # Record data
                t_sol = np.hstack((t_sol_ctl,time.time()-t0_lp))
                self.recorder.record(img,t_tr,u_act,x_ref,u_ref,x_est,x_ext,adv,t_sol)
            else:
                # trajectory has ended → HOLD Position
                self.policy_duration       = t_tr
                self.hold_prompt           = self.prompt
                self.spin_ready            = False
                self.spin_thread_started   = False
                self.prompt_thread_started = False
                self.prompt_buffer         = None

                self.hold_state = x_est.copy()
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9                       # Record start time
                self.sm                    = StateMachine.HOLD
                print('Trajectory Finished → HOLDing Position')
#NOTE deprecated
                # self._policy_duration = t_tr
                # self.sm = StateMachine.LAND
                # print('Trajectory Finished.')
        elif self.sm == StateMachine.HOLD:
            t0_lp  = time.time()                                                # Algorithm start time
            t_tr = self.get_current_trajectory_time()                           # Current trajectory time

            if t_tr < 5.0:
                # zch.publish_position_hold(self.get_current_timestamp_time(), self.hold_state, self.trajectory_setpoint_publisher)
                zch.publish_velocity_hold(
                    self.get_current_timestamp_time(),
                    self.trajectory_setpoint_publisher
                )
                # t_sol = np.hstack((time.time()-t0_lp,time.time()-t0_lp))    
                # self.recorder.record(img,t_tr,[0.0,0.0,0.0,0.0],self.hold_state,u_ref,x_est,x_ext,adv,t_sol)
#FIXME            
            # else:
            #     # time to flip into SPIN
            #     self.t_tr0 = self.get_clock().now().nanoseconds/1e9                       # Record start time
            #     self.sm = StateMachine.SPIN
            #     print('Trajectory Finished → SPINning to acquire...')
            else:
                self.sm = StateMachine.LAND
                print('Trajectory Finished → Landing...')
#FIXME
            # if not self.spin_thread_started:
            #     self.k_rdy = 0
            #     self.spin_thread_started = True
            #     t0   = self.get_clock().now().nanoseconds / 1e9
            #     xyz  = x_est[0:3]
            #     yaw0 = self.quat_to_yaw(x_est[6:10])
            #     threading.Thread(
            #         target=self.build_yaw_qp,
            #         args=(xyz, yaw0, t0),
            #         daemon=True
            #     ).start()

            # if not self.prompt_thread_started:
            #     self.prompt_thread_started = True
            #     threading.Thread(
            #         target=self.wait_for_prompt,
            #         daemon=True
            #     ).start()
#FIXME
            # if self.spin_ready and self.prompt_buffer != self.hold_prompt and np.linalg.norm(x_est[6:10]-self.q0) < self.q0_tol and np.abs(x_est[2]-self.z0) < self.z0_tol:
            # if self.spin_ready and np.linalg.norm(x_est[6:10]-self.q0) < self.q0_tol and np.abs(x_est[2]-self.z0) < self.z0_tol:    
            #     # self.hold_prompt = self.prompt_buffer
            #     print("Query changed → SPINning to acquire...")
            #     self.t_tr0 = self.get_clock().now().nanoseconds / 1e9
            #     zch.engage_offboard_control_mode(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            #     self.sm = StateMachine.SPIN
#
        elif self.sm == StateMachine.SPIN:
            t_tr  = self.get_current_trajectory_time()

            if t_tr < 5.0:
                zch.publish_velocity_hold_with_yaw_rate(
                    self.get_current_timestamp_time(),
                    self.trajectory_setpoint_publisher,
                    self.vehicle_rates_setpoint_publisher,
                    0.2
                )
            else:
                self.spin_thread_started   = False
                self.prompt_thread_started = False
                self.prompt_buffer         = None
                self.t_tr0 = self.get_clock().now().nanoseconds/1e9                       # Record start time
                self.sm                    = StateMachine.HOLD
                print('Trajectory Finished → HOLDing Position')
#
        else:
            # State Actions
            zch.land(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            
            # print("Closing Vision Thread")
            # self.vision_shutdown = True
            # self.vision_thread.join(timeout=1.0)
            # self.vision_started = False
            # print("Vision Thread Offline")
            
            # # Close the camera (if exists)
            # if self.pipeline is not None:
            #     zch.close_camera(self.pipeline)

            # print("Closing Keyboard Thread")
            # self.kb_shutdown.set()
            # self.kb_thread.join(timeout=1.0)
            # print("Keyboard Thread Offline")

            # # Close the controller (if mpc)
            # if self.isNN is False:
            #     self.ctl.clear_generated_code()
                
            # Save data
            output_path = self.recorder.save()

            # Print statistics
            if output_path is not None:
                print('Trajectory Finished.')
                print('=====================================================================')
                
                # Compute and print statistics
                Tcmp = self.recorder.Tsol[:-1,:]
                Tcmd = self.recorder.Tsol[-1,:]

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
            
            # # Wait for GCS clearance
            input("Press Enter to close node...")
            print("Closing node...")
            self._cleanup()
            # self.destroy_node()
            # # try:
            # #     super().destroy_node()
            # # except Exception:
            # #     pass
            # self.destroy_node()
            rclpy.shutdown()
            # # sys.exit(0)
            # # exit()
            return

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
    # controller.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()

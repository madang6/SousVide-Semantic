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
import albumentations as A

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image

from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    VehicleOdometry,
    VehicleRatesSetpoint,
)

from sousvide.control.pilot import Pilot
import figs.control.vehicle_rate_mpc as vr_mpc

import sousvide.flight.vision_preprocess_alternate as vp
import sousvide.flight.zed_command_helper as zch
import figs.dynamics.model_equations as me
import figs.dynamics.model_specifications as qc
import figs.utilities.trajectory_helper as th
import sousvide.visualize.plot_synthesize as ps
import sousvide.visualize.record_flight as rf
# import sousvide.visualize.plot_flight as pf

class StateMachine(Enum):
    """Enum class for pilot state."""
    INIT   = 0
    READY  = 1
    ACTIVE = 2
    LAND   = 3

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

        print("=====================================================================")
        print("---------------------------------------------------------------------")

        # ---------------------------------------------------------------------------------------
        # Some useful constants & variables -----------------------------------------------------
        # ---------------------------------------------------------------------------------------
        hz = 10

        # Camera Parameters
        cam_dim,cam_fps, = [376,672,3],30           # we use the convention: [height,width,channels]

        cb_sensor = ReentrantCallbackGroup()
        cb_control = ReentrantCallbackGroup()

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
        qos_cam = QoSProfile(
             reliability=ReliabilityPolicy.BEST_EFFORT,
             durability=DurabilityPolicy.VOLATILE,
             history=HistoryPolicy.KEEP_LAST,
             depth=1
         )

        # subscribe to raw images—depth=1 keeps only latest frame
        self.create_subscription(
            Image,
            'camera/image_raw',
            self._on_image,
            qos_cam,
            callback_group=cb_sensor
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
        else:
            self.isNN = True
            self.ctl = Pilot(mission_config["cohort"],mission_config["pilot"])
            print(f"Using Pilot: {mission_config['pilot']}")
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

        # Initialize CLIPSeg Model
        self.prompt = mission_config.get('prompt', '')
        print(f"Query Set to: {self.prompt}")
        self.hf_model = mission_config.get('hf_model', 'CIDAS/clipseg-rd64-refined')
        self.onnx_model_path = mission_config.get('onnx_model_path')

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
            self.Tpi = np.arange(0.0, 15.0 + dt, dt)
            self.obj = np.zeros((18,1))
            self.q0 = np.array([0.0, 0.0, -0.70710678, 0.70710678])          # 
            self.z0 = -0.50                                    # init altitude offset
            self.xref = np.zeros(10)                          # no reference state
            self.uref = -6.90*np.ones((4,))                   # reference input
            record_duration = self.Tpi[-1] + self.t_lg        # 


        # Initialize control variables
#NOTE commented out variables don't exist for RRT* generated tXUi
        self.node_start = False                                 # Node start flag
        self.node_status_informed = False                       # Node status informed flag

        self.drone_config = drone_config                        # drone configuration
        self.k_rdy = 0                                          # ready counter
        self.t_tr0 = 0.0                                        # trajectory start time
        self.vo_est = VehicleOdometry()                         # current state vector (estimated)
        self.vo_ext = VehicleOdometry()                         # current state vector (external)
        self.vrs = VehicleRatesSetpoint()                       # current vehicle rates setpoint
        self.znn = (torch.zeros(self.ctl.model.Nz)              # current visual feature vector
                    if isinstance(self.ctl,Pilot) else None)   

        # State Machine and Failsafe variables
        self.sm = StateMachine.INIT                                    # state machine
        
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
        # self.cmdLoop = self.create_timer(1/self.ctl.hz, self.commander)
        self.cmdLoop = self.create_timer(
            1.0/self.ctl.hz,
            self.commander,
            callback_group=cb_control
        )

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

        # Wait for GCS clearance
        input("Press Enter to proceed...")

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
    
    def on_image_callback(self, msg: Image):
        """Grab & run inference in its own thread."""
        imgz, _ = zch.get_image(self.pipeline)
        if imgz is None:
            return
        frame = cv2.cvtColor(imgz, cv2.COLOR_BGR2RGB)
        # store last image (non-blocking for control)
        self.latest_mask = self.vision_model.clipseg_hf_inference(
            frame, self.prompt,
            resize_output_to_input=True,
            use_refinement=False,
            use_smoothing=False,
            scene_change_threshold=1.0,
            verbose=False
        )

    def commander(self) -> None:
        """Main control loop for generating commands."""
        # Looping Callback Actions
        x_est,u_pr = zch.vo2x(self.vo_est,self.T_e2w)[0:10],zch.vrs2uvr(self.vrs)
        x_ext,znn,obj = zch.vo2x(self.vo_ext)[0:10],self.znn,self.obj
        x_est[6:10] = th.obedient_quaternion(x_est[6:10],self.xref[6:10])
        x_ext[6:10] = th.obedient_quaternion(x_ext[6:10],self.xref[6:10])

#FIXME: 
        img = getattr(self, 'latest_mask', None)       # grab freshest available
        
        # imgz,_ = zch.get_image(self.pipeline)
        # if imgz is None:
        #     self.sm = StateMachine.LAND
        #     print('Camera feed lost. Initiating landing sequence.')

        # # Process Image
        # frame_rgb = cv2.cvtColor(imgz, cv2.COLOR_BGR2RGB)
        
        # # Inference
        # t0 = time.time()
        # img = self.vision_model.clipseg_hf_inference(
        #     frame_rgb,
        #     self.prompt,
        #     resize_output_to_input=True,
        #     use_refinement=False,
        #     use_smoothing=False,
        #     scene_change_threshold=1.0,
        #     verbose=False,
        # )
        

        zch.heartbeat_offboard_control_mode(self.get_current_timestamp_time(),self.offboard_control_mode_publisher)

        # State Machine
        if self.sm == StateMachine.INIT:
            # Looping State Actions

            # Check if we have state information
            if self.vo_est.timestamp > 0:
                # State Actions
                self.T_e2w[0:3,0:3] = R.from_quat(zch.vo2x(self.vo_est)[6:10]).as_matrix().T

                self.k_rdy = 0
                self.sm = StateMachine.READY

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
                self._policy_duration = t_tr
                self.sm = StateMachine.LAND
                print('Trajectory Finished.')
        else:
            # State Actions
            zch.land(self.get_current_timestamp_time(),self.vehicle_command_publisher)
            
            # Close the camera (if exists)
            if self.pipeline is not None:
                zch.close_camera(self.pipeline)

            # Close the controller (if mpc)
            if self.isNN is False:
                self.ctl.clear_generated_code()
                
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
                print(f"Policy was active for {self._policy_duration:.2f} seconds.")
                print("Policy Component Compute Time (s): ", mu_t_cmp)
                print("Policy Total Compute Frequency (hz):", hz_cmp)
                print("Total Command Loop Frequency (hz): ", hz_cmd)
                print("Controller Minimum Frequency (hz): ", hz_min)
                print('=====================================================================')
            
            # Wait for GCS clearance
            input("Press Enter to close node...")
            print("Closing node...")
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
    executor = MultiThreadedExecutor()
    executor.add_node(controller)
    executor.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
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



# ---------- Viz helpers ----------


def ensure_hw_match(reference_hw: tuple[int, int], arr: np.ndarray) -> np.ndarray:
    H, W = reference_hw
    if arr.shape[:2] != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    return arr


# ---------- Main script ----------


def parse_args():
    p = argparse.ArgumentParser()
    # Camera init
    p.add_argument("--cam-width",  type=int, default=None)
    p.add_argument("--cam-height", type=int, default=None)
    p.add_argument("--cam-fps",    type=int, default=None)
    # CLIPSeg
    p.add_argument("--prompt", type=str, default="boxes")
    p.add_argument("--hf-model", type=str, default="CIDAS/clipseg-rd64-refined")
    p.add_argument("--onnx-model-path", type=str, default=None)
    p.add_argument("--onnx-model-fp16-path", type=str, default=None)
    p.add_argument("--mode", choices=["full", "depth_only"], default="full",
                   help="full = CLIPSeg+range+3 videos, depth_only = capture XYZ+VIEW.DEPTH, print center range, save depth video only")
    p.add_argument("--range-source", choices=["xyz", "z"], default="xyz",
                   help="For depth_only: use 'xyz' for Euclidean range (MEASURE.XYZ) or 'z' for optical-axis depth (MEASURE.DEPTH)")
    # Estimator
    p.add_argument("--top-percent", type=float, default=10.0)
    p.add_argument("--min-pixels",  type=int, default=200)
    p.add_argument("--z-min",  type=float, default=0.3)
    p.add_argument("--z-max",  type=float, default=20.0)
    # Saving
    p.add_argument("--input-video-path", dest="input_video_path", type=str, default=None)
    p.add_argument("--out-clipseg", type=str, default="clipseg_out.mp4")
    p.add_argument("--out-depth",   type=str, default="depth_out.mp4")
    p.add_argument("--out-sideby",  type=str, default="combo_out.mp4")
    p.add_argument("--save-fps",    type=float, default=15.0)
    # Loop
    p.add_argument("--rate-hz",     type=float, default=15.0)   # encode rate & print throttle
    p.add_argument("--max-seconds", type=float, default=None)   #
    return p.parse_args()

def pick(val, cfg, key, default=None):
    return val if val is not None else cfg.get(key, default)


class TestBaseline(Node):
    """Node for generating commands from SFTI."""

    def __init__(self,mission_name:str) -> None:
        super().__init__()

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


        #NOTE streamline testing non-mpc pilots
        # non-MPC (neural) case: give everything a neutral default
        dt = 1.0 / hz                                     # hz (control rate)
        self.Tpi = np.arange(0.0, 1.0 + dt, dt)
        self.obj = np.zeros((18,1))
        self.q0 = np.array([0.0, 0.0, 0.70710678, 0.70710678])          # determines starting orientation for ACTIVE mode to launch
        self.z0 = -0.50                                   # init altitude offset
        self.xref = np.zeros(10)                          # no reference state
        self.uref = -6.90*np.ones((4,))                   # reference input
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


        # Create a timer to publish control commands
        self.cmdLoop = self.create_timer(dt, self.commander)

        # Earth to World Pose
        self.T_e2w = np.eye(4)

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
            imgz, depth_xyz, _, _ = zch.get_image(self.pipeline, use_depth=self.use_depth)
            if imgz is None:
                time.sleep(0.01)
                continue
            t0_img = time.time()
            frame = cv2.cvtColor(imgz, cv2.COLOR_BGR2RGB)
            mask, raw_similarity = self.vision_model.clipseg_hf_inference(
                frame,
                self.prompt,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False
            )
            self.latest_depth_xyz = depth_xyz
            self.latest_mask = mask
            self.latest_similarity = raw_similarity
            t_elap = time.time() - t0_img
            self.img_times.append(t_elap)


    def test_run(self) -> None:
        """Main control loop for generating commands."""
        # Looping Callback Actions
        x_est,u_pr = zch.vo2x(self.vo_est,self.T_e2w)[0:10],zch.vrs2uvr(self.vrs)
        x_ext,obj = zch.vo2x(self.vo_ext)[0:10],self.obj
        x_est[6:10] = th.obedient_quaternion(x_est[6:10],self.xref[6:10])
        x_ext[6:10] = th.obedient_quaternion(x_ext[6:10],self.xref[6:10])

        img = self.latest_mask
        similarity = self.latest_similarity
        depth = self.latest_depth_xyz

        # print(f"Capturing live for {duration:.1f}s at {fps_cam} FPS…")
        # t_start = time.time()

        # try:
            # while (time.time() - t_start) < duration:
        # Grab LEFT + DEPTH (meters) with your updated get_image

        # Make sure depth matches overlay size (nearest)
        H, W = overlay_rgb.shape[:2]
        xyz_np = ensure_hw_match((H, W), xyz_np)

        t0 = time.time()
        ok, p_cam, uv, mask = zch.pose_from_similarity_xyz(
            similarity, xyz_np, top_percent=top_percent, min_pixels=min_pixels
        )
        tf = time.time()

        # Build frames for saving (imageio expects RGB)
        clipseg_rgb = overlay_rgb

        # Convert ZED VIEW.DEPTH (8-bit display) to RGB for saving, and match overlay size
        H, W = clipseg_rgb.shape[:2]
        depth_rgb = vp.depth_display_to_rgb(depth_viz, target_hw=(H, W))

        # Optional label
        # depth_rgb = depth_rgb.copy()
        # label = f"Depth (display)  Range est: {range_m:.2f} m" if ok and np.isfinite(range_m) else "Depth (display)  Range est: N/A"
        # cv2.putText(depth_rgb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,10), 3, cv2.LINE_AA)
        # cv2.putText(depth_rgb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 1, cv2.LINE_AA)

        # Append for imageio
        frames_clipseg.append(clipseg_rgb)
        frames_depth.append(depth_rgb)
        frames_combo.append(np.hstack([clipseg_rgb, depth_rgb]))
        frame_count += 1

        # Optional console
        if frame_count % max(1, int(fps_cam/2)) == 0:
            print(
                f"[{frame_count:04d}] "
                f"[TIME] Capture={t1-t0:.2f} ms | "
                f"[POSE] Pixel centroid=({uv[0]},{uv[1]}) | "
                f"Camera frame (X,Y,Z) = ({p_cam[0]:.2f}, {p_cam[1]:.2f}, {p_cam[2]:.2f}) m | "
            )


def main() -> None:
    args = parse_args()

    # Load JSON config (same path you already use)
    CONFIG_PATH = os.path.expanduser(
        "~/StanfordMSL/SousVide-Semantic/configs/perception/onnx_benchmark_config.json"
    )
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    # Merge settings (CLI > cfg > hard default)
    mode          = pick(args.mode,          cfg, "mode",           "full")
    range_source  = pick(args.range_source,  cfg, "range_source",   "xyz")
    width         = pick(args.cam_width,  cfg, "camera_width",   640)
    height        = pick(args.cam_height, cfg, "camera_height",  480)
    fps_cam       = pick(args.cam_fps,    cfg, "camera_fps",     30)
    duration      = pick(args.max_seconds, cfg, "camera_duration", 10.0)
    save_fps      = pick(args.save_fps,      cfg, "save_fps",       10.0)

    input_video_path = pick(args.input_video_path, cfg, "input_video_path",
                            "~/StanfordMSL/SousVide-Semantic/notebooks/out/live_placeholder.mp4")
    input_video_path = os.path.expanduser(input_video_path)
    video_dir = os.path.dirname(input_video_path)
    _, ext = os.path.splitext(os.path.basename(input_video_path))
    out_clipseg = os.path.join(video_dir, f"live_range_clipseg{ext}")
    out_depth   = os.path.join(video_dir, f"live_range_depth{ext}")
    out_combo   = os.path.join(video_dir, f"live_range_combo{ext}")

    # Range estimator params
    top_percent = float(cfg.get("range_top_percent", 10.0))
    min_pixels  = int(cfg.get("range_min_pixels", 200))

    frames_clipseg = []
    frames_depth   = []
    frames_combo   = []
    times = []
    frame_count = 0

    rclpy.init()
    test_run = TestBaseline()
    rclpy.spin(test_run)
    test_run.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
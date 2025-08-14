#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import pyzed.sl as sl
from rclpy.publisher import Publisher

from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    VehicleOdometry,
    VehicleRatesSetpoint,
    TrajectorySetpoint,
    ActuatorMotors
)

def euclidean_range_from_similarity(
    similarity: np.ndarray,
    xyz_m: np.ndarray,               # HxWx3 or HxWx4 from sl.MEASURE.XYZ
    top_percent: float = 10.0,
    min_pixels: int = 200,
    max_abs_coord: float = 1e3,      # clip XYZ to ±1000 m to avoid overflow
    r_min: float = 0.05,
    r_max: float = 200.0,
) -> tuple[bool, float]:
    if similarity is None or xyz_m is None or xyz_m.ndim != 3 or xyz_m.shape[2] < 3:
        return False, float('nan')

    # Use only X,Y,Z (ignore 4th channel if present)
    xyz = xyz_m[..., :3].astype(np.float32, copy=False)

    # Ensure shapes match (nearest keeps pixel alignment)
    Hs, Ws = similarity.shape[:2]
    Hx, Wx = xyz.shape[:2]
    if (Hs, Ws) != (Hx, Wx):
        xyz = cv2.resize(xyz, (Ws, Hs), interpolation=cv2.INTER_NEAREST)

    sim = similarity.astype(np.float32, copy=False)

    # Top-P% threshold without sorting whole array
    n = sim.size
    k = max(0, min(n - 1, int((1.0 - top_percent / 100.0) * n)))
    thr = np.partition(sim.ravel(), k)[k]

    # Gather ROI XYZ
    roi = xyz[sim >= thr]                    # shape (N, 3)
    if roi.size < 3 * max(10, min_pixels):
        return False, float('nan')

    # Filter out non-finite rows
    finite = np.all(np.isfinite(roi), axis=1)
    roi = roi[finite]
    if roi.shape[0] < min_pixels:
        return False, float('nan')

    # Clip absurd magnitudes to avoid overflow when squaring
    np.clip(roi, -max_abs_coord, max_abs_coord, out=roi)

    # Stable Euclidean norm (float64 + hypot chaining)
    x = roi[:, 0].astype(np.float64, copy=False)
    y = roi[:, 1].astype(np.float64, copy=False)
    z = roi[:, 2].astype(np.float64, copy=False)
    dist = np.hypot(np.hypot(x, y), z)

    # Keep plausible ranges
    dist = dist[(dist > r_min) & (dist < r_max)]
    if dist.size < min_pixels:
        return False, float('nan')

    return True, float(np.nanmedian(dist))

def publish_velocity_toward_range(
    timestamp: int,
    current_yaw_rad: float,
    current_range_m: float,
    target_range_m: float,
    traj_sp_pub: Publisher,
    kp: float = 0.6,
    vmax: float = 0.6,
    deadband_m: float = 0.05
) -> None:
    """Forward/back velocity along heading to close range error."""
    if not np.isfinite(current_range_m):
        publish_velocity_hold(timestamp, traj_sp_pub)
        return

    err = float(current_range_m - target_range_m)  # >0: too far
    vx_body = 0.0 if abs(err) <= deadband_m else float(np.clip(kp * err, -vmax, vmax))

    c, s = np.cos(current_yaw_rad), np.sin(current_yaw_rad)
    vx_ned = vx_body * c
    vy_ned = vx_body * s

    ts = TrajectorySetpoint(timestamp=timestamp)
    ts.velocity     = [vx_ned, vy_ned, 0.0]
    ts.position     = [float('nan')]*3
    ts.acceleration = [float('nan')]*3
    ts.jerk         = [float('nan')]*3
    ts.yaw          = float('nan')
    ts.yawspeed     = float('nan')
    traj_sp_pub.publish(ts)

def get_camera(height: int, width: int, fps: int) -> sl.Camera:
    """Initialize the ZED camera."""
    print("Initializing ZED camera if available...")

    try:
        camera = sl.Camera()
        
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA  # Automatically set resolution
        init_params.camera_fps = fps  # Set the camera FPS

        err = camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Error: {repr(err)}. Exiting program.")
            exit()

        print("Camera found!!!")
        return camera

    except Exception as e:
        print(f"No camera found: {e}")
        return None

def get_image(Camera: sl.Camera,
              use_depth: bool = False) -> tuple[np.ndarray, int]:
    """Capture an image from the ZED camera."""
    if Camera is None:
        print("Camera is not initialized.")
        return None, None

    runtime_parameters = sl.RuntimeParameters()

    # Grab a frame from the camera
    if Camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image = sl.Mat()
        Camera.retrieve_image(image, sl.VIEW.LEFT)
        if use_depth:
            depth = sl.Mat()
            depth_for_display = sl.Mat()
            # Camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
            Camera.retrieve_measure(depth, sl.MEASURE.XYZ)
            Camera.retrieve_image(depth_for_display, sl.VIEW.DEPTH)

        timestamp = Camera.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get timestamp

        # print(f"Image resolution: {image.get_width()} x {image.get_height()} || Image timestamp: {timestamp.get_milliseconds()}")

        return image.get_data(), depth.get_data(), depth_for_display.get_data(), timestamp.get_milliseconds()

    else:
        print("Failed to capture image.")
        return None, None, None, None

def close_camera(Camera: sl.Camera) -> None:
    """Close the ZED camera."""
    if Camera is not None:
        print('camera is closed')
        Camera.close()
    else:
        print("No camera closed")
    
def vo2x(vo:VehicleOdometry,T_12:np.ndarray=None) -> np.ndarray:
    """Convert VehicleOdometry to state vector in world frame."""

    xv = np.array([vo.position[0],vo.position[1],vo.position[2],
                    vo.velocity[0],vo.velocity[1],vo.velocity[2],
                    vo.q[1],vo.q[2],vo.q[3],vo.q[0],
                    vo.angular_velocity[0],vo.angular_velocity[1],vo.angular_velocity[2]])
    
    xv[6:10] += np.array([0.0,0.0,0.0,1e-9])
    xv[6:10] = xv[6:10]/np.linalg.norm(xv[6:10])

    # Transform to world frame
    if T_12 is not None:
        xv = x_transform(xv,T_12)

    return xv

def x_transform(x1:np.ndarray,T_12:np.ndarray) -> np.ndarray:
    """ Transform state vector given transform matrix."""

    # Variables
    R_12,t_12 = T_12[0:3,0:3],T_12[0:3,3]
    x2 = np.zeros(13)
    R_b1 =  R.from_quat(x1[6:10]).as_matrix()

    # Transform
    x2[0:3] = R_12@x1[0:3] + t_12
    x2[3:6] = R_12@x1[3:6]
    x2[6:10] = R.from_matrix(R_12@R_b1).as_quat()
    
    return x2

def am2unf(am:ActuatorMotors) -> np.ndarray:
    """Convert actuator motors to normalized motor force input."""
    return np.array([am.control[0],am.control[1],am.control[2],am.control[3]])

def vrs2uvr(vr:VehicleRatesSetpoint) -> np.ndarray:
    """Convert vehicle rates setpoint to vehicle rates input."""
    return np.array([vr.thrust_body[2],vr.roll,vr.pitch,vr.yaw])


# def publish_position_hold(timestamp: int,
#                              x_est: np.ndarray,
#                              traj_sp_pub) -> None:
#     """
#     Hold current position & heading using TrajectorySetpoint only.
#     - x_est: 13-element state [x,y,z, vx,vy,vz, qx,qy,qz,qw, ...]
#     - traj_sp_pub: Publisher<TrajectorySetpoint>
#     """
#     sp = TrajectorySetpoint(timestamp=timestamp)

#     # set position [x, y, z]
#     sp.position[0] = x_est[0]
#     sp.position[1] = x_est[1]
#     sp.position[2] = x_est[2]

#     # extract yaw from quaternion (qx,qy,qz,qw in x_est[6..9])
#     qx, qy, qz, qw = x_est[6], x_est[7], x_est[8], x_est[9]
#     sp.yaw = np.arctan2(2*(qw*qz + qx*qy),
#                         1 - 2*(qy*qy + qz*qz))

#     # ignore velocity/acceleration/jerk feed-forward
#     sp.velocity = [float('nan')] * 3
#     sp.acceleration = [float('nan')] * 3
#     sp.jerk = [float('nan')] * 3

#     traj_sp_pub.publish(sp)

def publish_velocity_hold(timestamp: int, traj_sp_pub) -> None:
    ts = TrajectorySetpoint(timestamp=timestamp)
    ts.velocity     = [0.0, 0.0, 0.0]     # hold zero velocity
    ts.position     = [float('nan')]*3
    ts.acceleration = [float('nan')]*3
    ts.jerk         = [float('nan')]*3
    ts.yaw          = float('nan')
    ts.yawspeed     = float('nan')
    traj_sp_pub.publish(ts)

def publish_velocity_hold_with_yaw_rate(timestamp: int,
                                        traj_sp_pub,
                                        rates_sp_pub,
                                        yaw_rate: float) -> None:
    # zero-velocity setpoint → holds X/Y/Z
    ts = TrajectorySetpoint(timestamp=timestamp)
    ts.velocity     = [0.0, 0.0, 0.0]
    ts.position     = [float('nan')] * 3
    ts.acceleration = [float('nan')] * 3
    ts.jerk         = [float('nan')] * 3
    ts.yaw          = float('nan')
    ts.yawspeed     = yaw_rate  # set yaw speed to desired rate
    traj_sp_pub.publish(ts)

    # # yaw-rate setpoint → spins at your chosen rate
    # vrs = VehicleRatesSetpoint(
    #     thrust_body = np.array([0.0, 0.0, float('nan')], dtype=np.float32),
    #     roll        = 0.0,
    #     pitch       = 0.0,
    #     yaw         = yaw_rate,
    #     timestamp   = timestamp
    # )
    # rates_sp_pub.publish(vrs)

# def publish_position_hold_with_yaw_rate(timestamp: int,
#                                            x_est: np.ndarray,
#                                            yaw_rate: float,
#                                            traj_sp_pub,
#                                            rates_sp_pub) -> None:
#     """
#     Hold x,y,z and command constant yaw rate.
#     - yaw_rate: radians/sec (+ = CCW looking down)
#     - traj_sp_pub:   Publisher<TrajectorySetpoint>
#     - rates_sp_pub:  Publisher<VehicleRatesSetpoint>
#     """
#     # publish hold-point TrajectorySetpoint
#     sp = TrajectorySetpoint(timestamp=timestamp)
#     sp.position[0] = x_est[0]
#     sp.position[1] = x_est[1]
#     sp.position[2] = x_est[2]
#     sp.yaw = float('nan')
#     sp.velocity = [float('nan')] * 3
#     sp.acceleration = [float('nan')] * 3
#     sp.jerk = [float('nan')] * 3
#     sp.yawspeed = float('nan')
#     traj_sp_pub.publish(sp)

#     # publish yaw-rate command
#     vrs = VehicleRatesSetpoint(
#         thrust_body=[0.0, 0.0, float('nan')].astype(np.float32),
#         roll=0.0,
#         pitch=0.0,
#         yaw=float(yaw_rate),
#         timestamp=timestamp
#     )
#     rates_sp_pub.publish(vrs)

def publish_uvr_command(timestamp:int,uvr:np.ndarray,vrs_publisher:Publisher) -> None:
    """Publish vehicle rates input (as a vehicle rates setpoint message)."""

    # Pack into message
    vrs = VehicleRatesSetpoint(
        thrust_body = np.array([0.0, 0.0, uvr[0]]).astype(np.float32),
        roll = float(uvr[1]),
        pitch = float(uvr[2]),
        yaw = float(uvr[3]),
        timestamp = timestamp
    )

    # Publish the message
    vrs_publisher.publish(vrs)

def engage_offboard_control_mode(timestamp:int,vc_publisher:Publisher) -> None:
    """Switch to offboard mode."""

    vehicle_command = VehicleCommand(
            command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,param2=6.0,param3=0.0,param4=0.0,param5=0.0,
            param6=0.0,param7=0.0,target_system=1,target_component=1,
            source_system=1,source_component=1,from_external=True,
            timestamp=timestamp
        )

    vc_publisher.publish(vehicle_command)

#NOTE: unused - requires global position estimate
def send_hold_mode(timestamp: int, vc_publisher: Publisher) -> None:
    """Switch PX4 into builtin autonomous HOLD (hover) mode."""
    vehicle_command = VehicleCommand(
        command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
        param1=1.0,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        param2=3.0,  # PX4_CUSTOM_MAIN_MODE_POSITION
        param3=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0,
        target_system=1, target_component=1,
        source_system=1, source_component=1, from_external=True,
        timestamp=timestamp
    )
    vc_publisher.publish(vehicle_command)

def land(timestamp:int,vc_publisher:Publisher) -> None:
    """Switch to land mode."""

    vehicle_command = VehicleCommand(
        command=VehicleCommand.VEHICLE_CMD_NAV_LAND,
        param1=0.0,param2=0.0,param3=0.0,param4=0.0,param5=0.0,
        param6=0.0,param7=0.0,target_system=1,target_component=1,
        source_system=1,source_component=1,from_external=True,
        timestamp=timestamp
    )

    vc_publisher.publish(vehicle_command)

def heartbeat_offboard_control_mode(
    timestamp: int,
    ocm_publisher: Publisher,
    *,
    body_rate: bool = False,
    velocity: bool  = False
) -> None:
    """Offboard heartbeat that can enable either body-rate or velocity control."""
    offboard_control_mode = OffboardControlMode(
        timestamp   = timestamp,
        position    = False,         
        velocity    = velocity,      # control mode for instruments-only
        acceleration= False,
        attitude    = False,
        body_rate   = body_rate,     # default control mode for Sous Vide
        actuator    = False
    )
    ocm_publisher.publish(offboard_control_mode)

# def heartbeat_offboard_control_mode(timestamp:int,ocm_publisher:Publisher) -> None:
#     """Send offboard heartbeat message."""

#     offboard_control_mode = OffboardControlMode(
#         timestamp = timestamp,
#         position = False,velocity = False,acceleration = False,
#         attitude = False,body_rate = True,actuator = False
#     )

#     ocm_publisher.publish(offboard_control_mode)

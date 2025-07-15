#!/usr/bin/env python3

from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyzed.sl as sl
from rclpy.publisher import Publisher

from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    VehicleOdometry,
    VehicleRatesSetpoint,
    ActuatorMotors
)

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

def get_image(Camera: sl.Camera) -> tuple[np.ndarray, int]:
    """Capture an image from the ZED camera."""
    if Camera is None:
        print("Camera is not initialized.")
        return None, None

    runtime_parameters = sl.RuntimeParameters()

    # Grab a frame from the camera
    if Camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image = sl.Mat()
        Camera.retrieve_image(image, sl.VIEW.LEFT)

        timestamp = Camera.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get timestamp

        # print(f"Image resolution: {image.get_width()} x {image.get_height()} || Image timestamp: {timestamp.get_milliseconds()}")

        return image.get_data(), timestamp.get_milliseconds()

    else:
        print("Failed to capture image.")
        return None, None

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
    
def heartbeat_offboard_control_mode(timestamp:int,ocm_publisher:Publisher) -> None:
    """Send offboard heartbeat message."""

    offboard_control_mode = OffboardControlMode(
        timestamp = timestamp,
        position = False,velocity = False,acceleration = False,
        attitude = False,body_rate = True,actuator = False
    )

    ocm_publisher.publish(offboard_control_mode)
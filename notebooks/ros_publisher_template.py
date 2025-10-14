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

        # Create a timer to publish control commands
        self.cmdLoop = self.create_timer(1/self.ctl.hz, self.commander)

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
    
    def commander(self) -> None:
        """Main control loop for generating commands."""

        """
        PLACE CALLS TO USEFUL CODE HERE!!!!
        """

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
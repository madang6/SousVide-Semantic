#!/usr/bin/env python3
import os, sys, signal, atexit, termios, tty, threading, argparse, time, json
from enum import IntEnum
from typing import Optional

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import UInt8, String, Bool
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import math

from cv_bridge import CvBridge

# import sousvide.flight.vision_preprocess_groundedsam as vpg


class HelperState(IntEnum):
    COMPLETE = 3  # we only emit COMPLETE


class OffboardHelper(Node):
    """
    Minimal offboard helper (no vision):
      - Keyboard line-edit to stage a query (only while edits are unlocked)
      - On handshake rising edge: publish COMPLETE + query (once), lock edits
      - On allow_edit=True message: unlock edits so user can change query
    """

    def __init__(self, drone_prefix: str, hz: int, mission_name: Optional[str] = None) -> None:
        super().__init__('offboard_helper_node')

        # --- config load kept (optional), but we won't use vision values ---
        cfg = {}
        if mission_name:
            workspace_path = os.path.dirname(os.path.dirname(__file__))
            mission_path = os.path.join(workspace_path, "configs", "missions", f"{mission_name}.json")
            if os.path.isfile(mission_path):
                with open(mission_path, 'r') as f:
                    cfg = json.load(f)
                self.get_logger().info(f"Loaded mission config: {mission_path}")
            else:
                raise FileNotFoundError(f"Mission file not found: {mission_path}")

        self.drone_prefix = drone_prefix
        self.hz = hz

        # CHANGE: edit lock replaces “processing”/vision gating
        self.handshake_ok = False           # tracks latest handshake value
        self.edit_locked = False            # when True, keyboard is ignored
        self.reported = False               # publish-once latch per handshake window

        self.staging_query = ""             # live-edited when unlocked
        self.active_query  = ""             # frozen on handshake rising edge

        self.ui_mode = 'LISTEN'             # LISTEN or PROCESS (for your status line)
        self.ui_shutdown = threading.Event()
        self.ui_thread = threading.Thread(target=self._ui_loop, daemon=True)
        self.ui_thread.start()

        # --- keyboard setup unchanged ---
        self._orig_tty = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.kb_shutdown = threading.Event()
        self._kb_thread = threading.Thread(target=self._kb_loop, daemon=True)
        self._kb_thread.start()

        # --- QoS (keep as-is) ---
        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.qos_state = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Subscribers ---
        self.handshake_sub = self.create_subscription(
            Bool, f'{self.drone_prefix}/offboard_helper/handshake', self._handshake_cb, self.qos_state)

        # CHANGE: new “unlock” message
        self.allow_edit_sub = self.create_subscription(
            Bool, f'{self.drone_prefix}/offboard_helper/allow_edit', self._allow_edit_cb, self.qos_state)

        # --- Publishers ---
        self.state_cmd_pub = self.create_publisher(
            UInt8, f'{self.drone_prefix}/offboard_helper/state_cmd', self.qos_cmd)
        self.query_pub = self.create_publisher(
            String, f'{self.drone_prefix}/offboard_helper/query', self.qos_cmd)
        self.diag_pub = self.create_publisher(
            String, f'{self.drone_prefix}/offboard_helper/diag', self.qos_cmd)

        # --- Timer loop (now trivial) ---
        self.timer = self.create_timer(1.0 / self.hz, self._loop)

        self.get_logger().info('OffboardHelper (no vision) up. Type your query; ENTER commits while unlocked.')
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._sigint)

    # ===== CLI helpers =====
    def _c(self, s, color):  # simple colorizer
        colors = {
            'red':'\x1b[31m','green':'\x1b[32m','yellow':'\x1b[33m',
            'blue':'\x1b[34m','magenta':'\x1b[35m','cyan':'\x1b[36m',
            'bold':'\x1b[1m','reset':'\x1b[0m'
        }
        return f"{colors.get(color,'')}{s}{colors['reset']}"

    def _banner(self, title, color='cyan'):
        bar = '═' * max(10, len(title) + 2)
        print(f"\n\x1b[1m{self._c('╔'+bar+'╗', color)}\x1b[0m")
        print(f"\x1b[1m{self._c('║ '+title+' ║', color)}\x1b[0m")
        print(f"\x1b[1m{self._c('╚'+bar+'╝', color)}\x1b[0m")

    def _ui_loop(self):
        # hide cursor
        try: sys.stdout.write("\x1b[?25l"); sys.stdout.flush()
        except: pass
        spinner = ['|','/','-','\\']
        i = 0
        while not self.ui_shutdown.is_set():
            if self.ui_mode == 'PROCESS':  # This is our LOCKED state
                left = self._c('LOCKED', 'red')
                q = (self.active_query or '—')
                line = f"{left}  processing query='{q}'"
            else:  # LISTENING state
                left = self._c('LISTENING', 'yellow')
                q = (self.staging_query or '—')
                line = f"{left}  {spinner[i%4]}  staged_query='{q}'"
            i += 1
            # draw one-line status
            try:
                sys.stdout.write('\r' + ' ' * (os.get_terminal_size().columns - 1))
                sys.stdout.write('\r' + line[:os.get_terminal_size().columns - 1])
                sys.stdout.flush()
            except Exception:
                pass
            time.sleep(0.15)
        # show cursor again
        try: sys.stdout.write("\x1b[?25h\n"); sys.stdout.flush()
        except: pass

    # ---------- Callbacks ----------
    # CHANGE: handshake behavior now: rising edge => publish once & lock edits
    def _handshake_cb(self, msg: Bool):
        new_val = bool(msg.data)
        if new_val and not self.handshake_ok:
            # rising: freeze & send
            self.active_query = self.staging_query.strip()
            self.reported = False
            self.edit_locked = True
            self.ui_mode = 'PROCESS'
            self._banner(f"Handshake ON → sending query: '{self.active_query or '—'}'", color='green')
            self._publish_once_now()
            # Reset handshake_ok to False so we can detect the next rising edge
            self.handshake_ok = False
        elif not new_val and self.handshake_ok:
            # falling: keep locked; we wait for allow_edit to unlock
            self._banner("Handshake OFF", color='yellow')
            self.handshake_ok = new_val
        # Don't update handshake_ok for rising edge case since we reset it above

    # CHANGE: separate unlock signal
    def _allow_edit_cb(self, msg: Bool):
        if bool(msg.data):
            self.edit_locked = False
            self.reported = False
            self.ui_mode = 'LISTEN'
            self._banner("Allow-edit received → terminal unlocked", color='cyan')
            self._publish_diag('Ready for new query (ENTER to stage)')

    # CHANGE: one-shot publisher
    def _publish_once_now(self):
        if self.reported:
            return
        q = self.active_query
        self._publish_diag(f'Sending COMPLETE + query="{q}"')
        self._send_state_cmd(HelperState.COMPLETE)
        self.query_pub.publish(String(data=q))
        self.reported = True


    # ---------- Main loop ----------
    def _loop(self):
        return

    # ---------- Publisher helpers ----------
    def _publish_diag(self, message: str):
        """Publish diagnostic message"""
        self.diag_pub.publish(String(data=message))
        self.get_logger().info(f"DIAG: {message}")

    def _send_state_cmd(self, state: HelperState):
        """Send state command"""
        self.state_cmd_pub.publish(UInt8(data=int(state)))

    # ---------- Keyboard: line-edit only ----------
    # Keyboard loop: honor edit_locked instead of handshake gating
    def _kb_loop(self):
        prompt = "[query] "
        try:
            sys.stdout.write(prompt); sys.stdout.flush()
        except Exception:
            pass

        while not self.kb_shutdown.is_set():
            try:
                ch = sys.stdin.read(1)
                code = ord(ch)

                # ENTER = commit staged (only if unlocked)
                if code in (10, 13):
                    if not self.edit_locked:
                        self.staging_query = self.staging_query.strip()
                        self._publish_diag(f'Query staged: "{self.staging_query}"')
                    else:
                        self._publish_diag('Ignored ENTER: edits locked')
                    sys.stdout.write("\n" + prompt); sys.stdout.flush()
                    continue

                # ESC = clear staged (only if unlocked)
                if code == 27:
                    if not self.edit_locked:
                        self.staging_query = ""
                        self._publish_diag('Staged query cleared (ESC)')
                    else:
                        self._publish_diag('Ignored ESC: edits locked')
                    sys.stdout.write("\n" + prompt); sys.stdout.flush()
                    continue

                # BACKSPACE
                if code in (8, 127):
                    if not self.edit_locked and len(self.staging_query) > 0:
                        self.staging_query = self.staging_query[:-1]
                        sys.stdout.write("\b \b"); sys.stdout.flush()
                    continue

                # Printable ASCII
                if 32 <= code <= 126:
                    if not self.edit_locked:
                        self.staging_query += ch
                        sys.stdout.write(ch); sys.stdout.flush()
                    # else ignore typing
                    continue

            except Exception:
                time.sleep(0.01)


    # ---------- Cleanup ----------
    def _cleanup(self):
        try:
            self.ui_shutdown.set()
            if getattr(self, 'ui_thread', None):
                self.ui_thread.join(timeout=0.5)
        except Exception:
            pass

        try:
            self.kb_shutdown.set()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_tty)
        except Exception:
            pass
        

    def _sigint(self, signum, frame):
        self._cleanup()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Offboard helper node')
    parser.add_argument('--drone', type=str, default='/drone0', help='Drone namespace/prefix (e.g., /drone0)')

    parser.add_argument('--hz', type=int, default=20, help='Main loop Hz')
    parser.add_argument('--mission', type=str, default='', help='Mission config name (without .json)')
    args = parser.parse_args()

    rclpy.init()
    # depth_topic = args.depth.strip() or None
    node = OffboardHelper(
        drone_prefix=args.drone,

        hz=args.hz,
        mission_name=(args.mission.strip() or None),
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

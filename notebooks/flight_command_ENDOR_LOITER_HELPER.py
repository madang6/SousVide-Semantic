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
from cv_bridge import CvBridge

import sousvide.flight.vision_preprocess_groundedsam as vpg


class HelperState(IntEnum):
    COMPLETE = 3  # we only emit COMPLETE


class OffboardHelper(Node):
    """
    Minimal offboard helper:
      - Keyboard line-edit to stage a query (only while not processing)
      - Handshake (Bool) from drone gates processing
      - Runs vision with frozen active_query during handshake window
      - If present=True (and optional proximity_ok), publishes:
          1) state_cmd = COMPLETE
          2) query = active_query
      - No other publishes
    """

    def __init__(
        self,
        drone_prefix: str,
        image_topic: str,
        depth_topic: Optional[str],
        hz: int,
        mission_name: Optional[str] = None,
    ) -> None:
        super().__init__('offboard_helper_node')

        # ---------- Config (optional mission file) ----------
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

        # ---------- Args / members ----------
        self.drone_prefix = drone_prefix
        self.image_topic = image_topic
        self.depth_topic = depth_topic
        self.hz = hz

        self.bridge = CvBridge()

        self.handshake_ok = False
        self.staging_query = ""  # edited while not processing
        self.active_query = ""   # frozen during processing window
        self.reported = False    # ensures single publish per window

        self.latest_rgb = None
        self.latest_depth = None

        # ---------- Keyboard: line-edit only ----------
        self._orig_tty = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.kb_shutdown = threading.Event()
        self._kb_thread = threading.Thread(target=self._kb_loop, daemon=True)
        self._kb_thread.start()

        # ---------- QoS ----------
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

        # ---------- Vision model ----------
        self.get_logger().info("Initializing GroundedSAMHFModel...")
        gd_model_id   = cfg.get('gd_model_id',   'IDEA-Research/grounding-dino-tiny')
        sam_model_id  = cfg.get('sam_model_id',  'facebook/sam-vit-base')
        box_thr       = float(cfg.get('box_threshold', 0.35))
        text_thr      = float(cfg.get('text_threshold', 0.25))
        mask_iou_thr  = float(cfg.get('mask_iou_threshold', 0.0))
        overlay_alpha = float(cfg.get('overlay_alpha', 0.45))

        self.model = vpg.GroundedSAMHFModel(
            gd_model_id=gd_model_id,
            sam_model_id=sam_model_id,
            box_threshold=box_thr,
            text_threshold=text_thr,
            mask_iou_threshold=mask_iou_thr,
            overlay_alpha=overlay_alpha,
            return_patches=False,
        )

        # ---------- Subscribers ----------
        self.handshake_sub = self.create_subscription(
            Bool, f'{self.drone_prefix}/offboard_helper/handshake', self._handshake_cb, self.qos_state)

        self.rgb_sub = self.create_subscription(
            CompressedImage,
            f'{self.drone_prefix}/zed/zed_node/rgb/image_rect_color/compressed',
            self._rgb_compressed_cb,
            qos_profile_sensor_data
        )

        # ---------- Publishers ----------
        self.state_cmd_pub = self.create_publisher(
            UInt8, f'{self.drone_prefix}/offboard_helper/state_cmd', self.qos_cmd)
        self.query_pub = self.create_publisher(
            String, f'{self.drone_prefix}/offboard_helper/query', self.qos_cmd)
        self.diag_pub = self.create_publisher(
            String, f'{self.drone_prefix}/offboard_helper/diag', self.qos_cmd)

        # ---------- Loop timer ----------
        self.timer = self.create_timer(1.0 / self.hz, self._loop)

        self.get_logger().info('OffboardHelper up. Type your query (only effective when handshake is OFF). ENTER commits.')

        # Cleanup
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._sigint)

    # ---------- Callbacks ----------
    def _handshake_cb(self, msg: Bool):
        new_val = bool(msg.data)
        # rising edge → freeze active query & reset publish latch
        if new_val and not self.handshake_ok:
            self.active_query = self.staging_query
            self.reported = False
            self._publish_diag(f'Handshake ON. Using active_query="{self.active_query}"')
        # falling edge → allow edits again
        if not new_val and self.handshake_ok:
            self.reported = False
            self._publish_diag('Handshake OFF. You can edit the query.')
        self.handshake_ok = new_val

    def _rgb_cb(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'RGB conversion failed: {e}')

    def _rgb_compressed_cb(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            self.latest_rgb = bgr

    def _depth_cb(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion failed: {e}')

    # ---------- Main loop ----------
    def _loop(self):
        if not self.handshake_ok:
            return
        if self.latest_rgb is None:
            self._publish_diag('Waiting for RGB...')
            return

        rgb = self.latest_rgb
        query = self.active_query  # frozen during this processing window

        try:
            overlay, scaled, present, extras = self.model.grounded_sam_hf_inference(
                rgb,
                query,
                resize_output_to_input=True,
                use_refinement=False,
                use_smoothing=False,
                scene_change_threshold=1.0,
                verbose=False,
            )
        except Exception as e:
            self._publish_diag(f'Vision error: {e}')
            return

        proximity_ok = True
        if self.latest_depth is not None:
            proximity_ok = self._simple_proximity_check(self.latest_depth, max_depth=0.6, frac=0.04)

        if present and proximity_ok and not self.reported:
            self._publish_diag(f'Found target for query="{query}" → sending COMPLETE + query')
            self._send_state_cmd(HelperState.COMPLETE)
            self.query_pub.publish(String(data=query))
            self.reported = True

        # (Optional) save artifacts:
        # if 'masks_u8' in extras:
        #     cv2.imwrite('/tmp/offboard_mask.png', extras['masks_u8'])

    # ---------- Helpers ----------
    def _simple_proximity_check(self, depth: np.ndarray, max_depth: float = 0.6, frac: float = 0.04) -> bool:
        d = depth.astype(np.float32)
        if d.mean() > 5.0:  # likely mm → convert to meters
            d = d / 1000.0
        mask = d > 0
        if mask.sum() == 0:
            return True
        frac_close = float(((d < max_depth) & mask).sum()) / float(mask.sum())
        return frac_close < frac

    def _send_state_cmd(self, next_state: HelperState):
        msg = UInt8()
        msg.data = int(next_state)
        self.state_cmd_pub.publish(msg)

    def _publish_diag(self, text: str):
        self.diag_pub.publish(String(data=text))

    # ---------- Keyboard: line-edit only ----------
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

                # ENTER commits staged edits (only when NOT processing)
                if code in (10, 13):
                    if not self.handshake_ok:
                        self.staging_query = self.staging_query.strip()
                        self._publish_diag(f'Query staged: "{self.staging_query}"')
                    else:
                        self._publish_diag('Ignored ENTER: processing active (query frozen)')
                    sys.stdout.write("\n" + prompt); sys.stdout.flush()
                    continue

                # ESC clears staged text (only when NOT processing)
                if code == 27:
                    if not self.handshake_ok:
                        self.staging_query = ""
                        self._publish_diag('Staged query cleared (ESC)')
                    else:
                        self._publish_diag('Ignored ESC: processing active (query frozen)')
                    sys.stdout.write("\n" + prompt); sys.stdout.flush()
                    continue

                # BACKSPACE
                if code in (8, 127):
                    if not self.handshake_ok and len(self.staging_query) > 0:
                        self.staging_query = self.staging_query[:-1]
                        sys.stdout.write("\b \b"); sys.stdout.flush()
                    continue

                # printable ASCII
                if 32 <= code <= 126:
                    if not self.handshake_ok:
                        self.staging_query += ch
                        sys.stdout.write(ch); sys.stdout.flush()
                    # else ignore typing during processing
                    continue

                # ignore others
            except Exception:
                time.sleep(0.01)

    # ---------- Cleanup ----------
    def _cleanup(self):
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
    parser.add_argument('--rgb', type=str, default='/zed/zed_node/rgb/image_rect_color', help='RGB image topic')
    parser.add_argument('--depth', type=str, default='', help='Depth image topic (optional)')
    parser.add_argument('--hz', type=int, default=20, help='Main loop Hz')
    parser.add_argument('--mission', type=str, default='', help='Mission config name (without .json)')
    args = parser.parse_args()

    rclpy.init()
    depth_topic = args.depth.strip() or None
    node = OffboardHelper(
        drone_prefix=args.drone,
        image_topic=args.rgb,
        depth_topic=depth_topic,
        hz=args.hz,
        mission_name=(args.mission.strip() or None),
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

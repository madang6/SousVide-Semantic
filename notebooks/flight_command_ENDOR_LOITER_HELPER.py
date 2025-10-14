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
        # image_topic: str,
        # depth_topic: Optional[str],
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
        # self.image_topic = image_topic
        # self.depth_topic = depth_topic
        self.hz = hz

        self.bridge = CvBridge()

        self.handshake_ok = False
        self.staging_query = ""  # edited while not processing
        self.active_query = ""   # frozen during processing window
        self.reported = False    # ensures single publish per window

        self.ui_mode = 'LISTEN'       # or 'PROCESS'
        self.recorded_frames = 0
        self.ui_shutdown = threading.Event()
        self.ui_thread = threading.Thread(target=self._ui_loop, daemon=True)
        self.ui_thread.start()

        self.proc_writer = None
        self.proc_out_path = None
        self.proc_fps = float(self.hz)   # use your main loop rate

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
        self.get_logger().info(f"Subscribing to COMPRESSED RGB: {self.rgb_sub}")

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
            if self.ui_mode == 'PROCESS':
                left = self._c('PROCESSING', 'green')
                rec  = self._c('● REC', 'red') if self.proc_writer else '   '
                q    = (self.active_query or '—')
                path = (self.proc_out_path or '')
                line = f"{left} {rec}  frames={self.recorded_frames}  query='{q}'  {path}"
            else:
                left = self._c('LISTENING', 'yellow')
                q    = (self.staging_query or '—')
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

    def _open_proc_video(self, frame_shape):
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ts = time.strftime('%Y%m%d_%H%M%S')
        safe_query = (self.active_query or 'noquery').replace(' ', '_')[:48]
        # self.proc_out_path = f'/tmp/processed_{ts}_{safe_query}.mp4'
        self.proc_out_path = f"/home/ewok/msl_users/maximilian/ssv/SousVide-Semantic/cohorts/processed_{ts}_{safe_query}.mp4"
        self.proc_writer = cv2.VideoWriter(self.proc_out_path, fourcc, self.proc_fps, (w, h))
        if not self.proc_writer.isOpened():
            self.proc_writer = None
            raise RuntimeError('Failed to open processed video writer')
        self._publish_diag(f'Processed video recording → {self.proc_out_path}')

    def _close_proc_video(self):
        if self.proc_writer is not None:
            self.proc_writer.release()
            self.proc_writer = None
            self._publish_diag(f'Processed video saved: {self.proc_out_path}')
            self.proc_out_path = None

    # ---------- Callbacks ----------
    def _handshake_cb(self, msg: Bool):
        new_val = bool(msg.data)
        # rising edge → freeze active query & reset publish latch
        if new_val and not self.handshake_ok:
            self.active_query = self.staging_query
            self.reported = False
            self.recorded_frames = 0
            self.ui_mode = 'PROCESS'
            self._banner(f"Handshake ON → using query: '{self.active_query or '—'}'", color='green')
        
        # falling edge → allow edits again
        if not new_val and self.handshake_ok:
            self.reported = False
            self.ui_mode = 'LISTEN'
            self._banner("Handshake OFF → stopped processing", color='yellow')
            self._close_proc_video()
        self.handshake_ok = new_val

    def _rgb_compressed_cb(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.latest_rgb = bgr

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

        # Record Mask:
        if overlay is not None:
            # open lazily on the first overlay of this window
            if self.proc_writer is None:
                try:
                    self._open_proc_video(overlay.shape)
                except Exception as e:
                    self._publish_diag(f'Processed video open error: {e}')
            if self.proc_writer is not None:
                ov = overlay
                if ov.dtype != np.uint8:
                    ov = np.clip(ov, 0, 255).astype(np.uint8)
                if ov.ndim == 2:
                    ov = cv2.cvtColor(ov, cv2.COLOR_GRAY2BGR)
                self.proc_writer.write(ov)
                self.recorded_frames += 1

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
            self._close_proc_video()
        except Exception:
            pass

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
    # parser.add_argument('--rgb', type=str, default='/zed/zed_node/rgb/image_rect_color', help='RGB image topic')
    # parser.add_argument('--depth', type=str, default='', help='Depth image topic (optional)')
    parser.add_argument('--hz', type=int, default=20, help='Main loop Hz')
    parser.add_argument('--mission', type=str, default='', help='Mission config name (without .json)')
    args = parser.parse_args()

    rclpy.init()
    # depth_topic = args.depth.strip() or None
    node = OffboardHelper(
        drone_prefix=args.drone,
        # image_topic=args.rgb,
        # depth_topic=depth_topic,
        hz=args.hz,
        mission_name=(args.mission.strip() or None),
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

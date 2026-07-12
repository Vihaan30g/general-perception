#!/usr/bin/env python3
"""
grounding_dino_node.py
Phase 1 of the perception pipeline (grounding dino -> sam -> pose).

- Subscribes to /zedx1/rgb (sensor_msgs/Image).
- Subscribes to /dino/active_prompt (std_msgs/String, TRANSIENT_LOCAL) for
  the current detection prompt. Starts empty -> produces zero boxes until
  a prompt is set via prompt_manager_node + set_prompt.py.
- Runs inference in a background thread using "latest-frame-wins": if the
  model is slower than the camera publish rate, stale frames are simply
  skipped rather than queued up (see prior architecture discussion).
- Displays annotated frames in an OpenCV window on the MAIN thread only
  (cv2 GUI calls must stay on one consistent thread) - press 'q' to quit,
  that is the only way this window/node closes besides normal shutdown.
- Publishes detections as JSON on /dino/detections, keyed to the source
  frame's header stamp, for a future SAM node to pick up and pair with the
  matching /zedx1/rgb frame.
"""

import json
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from dino.dino_engine import DinoEngine


@dataclass
class Frame:
    rgb: np.ndarray
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    seq: int


class LatestFrame:
    """Thread-safe 'most recent frame' handoff between the fast ROS callback
    and the slower inference loop. Older frames are overwritten, never
    queued."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._last_consumed_seq = -1

    def set(self, rgb, stamp_sec, stamp_nanosec, frame_id):
        with self._lock:
            self._seq += 1
            self._frame = Frame(rgb, stamp_sec, stamp_nanosec, frame_id, self._seq)

    def get_if_new(self) -> Optional[Frame]:
        with self._lock:
            if self._frame is None or self._frame.seq == self._last_consumed_seq:
                return None
            self._last_consumed_seq = self._frame.seq
            return self._frame


class GroundingDinoNode(Node):
    def __init__(self):
        super().__init__("grounding_dino_node")
        self.bridge = CvBridge()
        self.latest = LatestFrame()

        self.prompt_lock = threading.Lock()
        self.current_prompt = ""

        self.engine = DinoEngine()
        self.get_logger().info("Loading Grounding DINO model (this can take a while)...")
        self.engine.load()
        self.get_logger().info("Model loaded.")

        # Match Isaac Sim's typical camera bridge QoS. If you get zero
        # messages, check `ros2 topic info /zedx1/rgb --verbose` - if the
        # publisher is actually RELIABLE, switch this to match.
        rgb_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, "/zedx1/rgb", self._on_rgb, rgb_qos)

        prompt_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(String, "/dino/active_prompt", self._on_prompt, prompt_qos)

        self.detections_pub = self.create_publisher(String, "/dino/detections", 10)

        self._last_annotated = None
        self._annotated_lock = threading.Lock()

        self._stop = threading.Event()
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

    # -- ROS callbacks: kept intentionally minimal, no heavy work here -----
    def _on_rgb(self, msg: Image):
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.latest.set(rgb, msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id)

    def _on_prompt(self, msg: String):
        with self.prompt_lock:
            self.current_prompt = msg.data.strip()
        self.get_logger().info(f"Prompt updated: '{self.current_prompt}'")

    def _get_prompt(self) -> str:
        with self.prompt_lock:
            return self.current_prompt

    # -- background inference thread ---------------------------------------
    def _inference_loop(self):
        while not self._stop.is_set():
            frame = self.latest.get_if_new()
            if frame is None:
                time.sleep(0.005)
                continue

            prompt = self._get_prompt()
            detections: List[Dict] = []
            if prompt:
                try:
                    detections = self.engine.infer(frame.rgb, prompt)
                except Exception as e:
                    self.get_logger().error(f"DINO inference failed, skipping this frame: {e}")
                    time.sleep(0.05)
                    continue
            # if prompt is empty, detections stays [] -> no boxes drawn/published

            annotated = self._draw_boxes(frame.rgb, detections)
            with self._annotated_lock:
                self._last_annotated = annotated

            self._publish_detections(frame, prompt, detections)

    def _draw_boxes(self, rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{det["label"]} {det["score"]:.2f}'
            cv2.putText(img, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return img

    def _publish_detections(self, frame: Frame, prompt: str, detections: List[Dict]):
        payload = {
            "header": {
                "stamp_sec": frame.stamp_sec,
                "stamp_nanosec": frame.stamp_nanosec,
                "frame_id": frame.frame_id,
            },
            "prompt": prompt,
            "detections": detections,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.detections_pub.publish(msg)

    # -- display loop: MUST run on the main thread --------------------------
    def display_loop(self):
        window = "Grounding DINO"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while rclpy.ok() and not self._stop.is_set():
            with self._annotated_lock:
                img = self._last_annotated
            if img is not None:
                cv2.imshow(window, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' pressed - shutting down.")
                self._stop.set()
                break

            # Drain any pending ROS callbacks without blocking the display loop.
            rclpy.spin_once(self, timeout_sec=0.0)

        cv2.destroyAllWindows()

    def destroy_node(self):
        self._stop.set()
        self._infer_thread.join(timeout=1.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = GroundingDinoNode()
    try:
        node.display_loop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
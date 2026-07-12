#!/usr/bin/env python3
"""
sam_node.py
Phase 2 of the perception pipeline: SAM, prompted by Grounding DINO's boxes.

- Subscribes to /zedx1/rgb and keeps a small stamp-indexed cache of recent
  frames (kept INLINE here, not a separate module - avoids missing-import
  problems). Needed because DINO runs slower than the camera, so by the
  time a detections message arrives the live feed has moved on; we look up
  the exact matching frame by the stamp DINO already embeds in its output.
- Subscribes to /dino/detections (JSON string), latest-message-wins so a
  slow SAM never builds a backlog against DINO's own (already throttled)
  output rate.
- Displays a colored mask overlay in an OpenCV window; 'q' to quit.
- Publishes:
    /sam/instance_mask (sensor_msgs/Image, mono8): 0 = background,
        1..N = instance index, matching "instance_id" below.
    /sam/instances (std_msgs/String, JSON): per-instance label/score/box,
        for a future pose-estimation node.
"""

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from sam.sam_engine import SamEngine

FRAME_CACHE_MAX = 300  # ~2s of frames at 30fps - raise if SAM inference is slower than that


@dataclass
class Detections:
    stamp_key: tuple
    frame_id: str
    prompt: str
    detections: List[Dict] = field(default_factory=list)
    seq: int = 0


class LatestDetections:
    """Same latest-message-wins pattern used in grounding_dino_node: only
    the newest detections set is ever processed, older ones are dropped."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item: Optional[Detections] = None
        self._seq = 0
        self._last_consumed_seq = -1

    def set(self, item: Detections):
        with self._lock:
            self._seq += 1
            item.seq = self._seq
            self._item = item

    def get_if_new(self) -> Optional[Detections]:
        with self._lock:
            if self._item is None or self._item.seq == self._last_consumed_seq:
                return None
            self._last_consumed_seq = self._item.seq
            return self._item


class SamNode(Node):
    def __init__(self):
        super().__init__("sam_node")
        self.bridge = CvBridge()

        self.engine = SamEngine()
        self.get_logger().info("Loading SAM model (this can take a while)...")
        self.engine.load()
        self.get_logger().info("Model loaded.")

        # -- inline frame cache: stamp (sec, nanosec) -> rgb numpy array --
        self.frame_cache_lock = threading.Lock()
        self.frame_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

        self.latest_detections = LatestDetections()

        rgb_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, "/zedx1/rgb", self._on_rgb, rgb_qos)
        self.create_subscription(String, "/dino/detections", self._on_detections, 10)

        self.mask_pub = self.create_publisher(Image, "/sam/instance_mask", 10)
        self.instances_pub = self.create_publisher(String, "/sam/instances", 10)

        self._last_annotated = None
        self._annotated_lock = threading.Lock()

        self._stop = threading.Event()
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

    # -- ROS callbacks: kept minimal -----------------------------------------
    def _on_rgb(self, msg: Image):
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        with self.frame_cache_lock:
            self.frame_cache[key] = rgb
            while len(self.frame_cache) > FRAME_CACHE_MAX:
                self.frame_cache.popitem(last=False)  # evict oldest

    def _on_detections(self, msg: String):
        try:
            payload = json.loads(msg.data)
            print("got the file nigg! \n")    #DEBUG
        except json.JSONDecodeError:
            self.get_logger().warning("Bad JSON on /dino/detections, skipping.")
            return

        header = payload.get("header", {})
        key = (header.get("stamp_sec"), header.get("stamp_nanosec"))
        item = Detections(
            stamp_key=key,
            frame_id=header.get("frame_id", ""),
            prompt=payload.get("prompt", ""),
            detections=payload.get("detections", []),
        )
        self.latest_detections.set(item)

    def _lookup_frame(self, key: tuple) -> Optional[np.ndarray]:
        with self.frame_cache_lock:
            return self.frame_cache.get(key)

    # -- background inference thread -----------------------------------------
    def _inference_loop(self):
        while not self._stop.is_set():
            item = self.latest_detections.get_if_new()
            if item is None:
                time.sleep(0.005)
                continue

            rgb = self._lookup_frame(item.stamp_key)
            print("Found cached frame:", rgb is not None)
            if rgb is None:
                self.get_logger().debug(
                    f"No cached frame for stamp {item.stamp_key} - "
                    f"increase FRAME_CACHE_MAX if this happens often."
                )
                continue

            boxes = [d["box_xyxy"] for d in item.detections]
            labels = [d["label"] for d in item.detections]
            scores = [d["score"] for d in item.detections]

            try:
                masks = self.engine.infer(rgb, boxes) if boxes else []
            except Exception as e:
                self.get_logger().error(f"SAM inference failed, skipping this frame: {e}")
                time.sleep(0.05)
                continue
            


            print(rgb.shape)
            print(rgb.dtype)
            print(rgb.min(), rgb.max())

            annotated, instance_mask = self._build_outputs(rgb, masks, labels, scores)
            with self._annotated_lock:
                self._last_annotated = annotated

            self._publish(item, instance_mask, labels, scores, boxes)

    def _build_outputs(self, rgb, masks, labels, scores):
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        instance_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        rng = np.random.default_rng(42)  # stable colors across frames
        for i, mask in enumerate(masks):
            instance_id = i + 1  # 0 reserved for background
            instance_mask[mask] = instance_id

            color = rng.integers(0, 255, size=3).tolist()
            colored = np.zeros_like(overlay)
            colored[mask] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)

            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(overlay, f"{labels[i]} {scores[i]:.2f}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return overlay, instance_mask

    def _publish(self, item: Detections, instance_mask, labels, scores, boxes):
        mask_msg = self.bridge.cv2_to_imgmsg(instance_mask, encoding="mono8")
        mask_msg.header.stamp.sec = item.stamp_key[0] or 0
        mask_msg.header.stamp.nanosec = item.stamp_key[1] or 0
        mask_msg.header.frame_id = item.frame_id
        self.mask_pub.publish(mask_msg)

        instances = [
            {"instance_id": i + 1, "label": labels[i], "score": scores[i], "box_xyxy": boxes[i]}
            for i in range(len(labels))
        ]
        payload = {
            "header": {
                "stamp_sec": item.stamp_key[0],
                "stamp_nanosec": item.stamp_key[1],
                "frame_id": item.frame_id,
            },
            "prompt": item.prompt,
            "instances": instances,
        }
        out = String()
        out.data = json.dumps(payload)
        self.instances_pub.publish(out)

    # -- display loop: MUST run on the main thread ---------------------------
    def display_loop(self):
        window = "SAM"
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

            rclpy.spin_once(self, timeout_sec=0.0)

        cv2.destroyAllWindows()

    def destroy_node(self):
        self._stop.set()
        self._infer_thread.join(timeout=1.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = SamNode()
    try:
        node.display_loop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()




























# #!/usr/bin/env python3
# """
# sam_node.py
# Phase 2 of the perception pipeline: SAM, prompted by Grounding DINO's boxes.

# - Subscribes to /zedx1/rgb and keeps a small stamp-indexed cache of recent
#   frames (kept INLINE here, not a separate module - avoids missing-import
#   problems). Needed because DINO runs slower than the camera, so by the
#   time a detections message arrives the live feed has moved on; we look up
#   the exact matching frame by the stamp DINO already embeds in its output.
# - Subscribes to /dino/detections (JSON string), latest-message-wins so a
#   slow SAM never builds a backlog against DINO's own (already throttled)
#   output rate.
# - Displays a colored mask overlay in an OpenCV window; 'q' to quit.
# - Publishes:
#     /sam/instance_mask (sensor_msgs/Image, mono8): 0 = background,
#         1..N = instance index, matching "instance_id" below.
#     /sam/instances (std_msgs/String, JSON): per-instance label/score/box,
#         for a future pose-estimation node.
# """

# import json
# import threading
# import time
# from collections import OrderedDict
# from dataclasses import dataclass, field
# from typing import Optional, Dict, List

# import cv2
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from cv_bridge import CvBridge

# from sam.sam_engine import SamEngine

# FRAME_CACHE_MAX = 300  # ~2s of frames at 30fps - raise if SAM inference is slower than that


# @dataclass
# class Detections:
#     stamp_key: tuple
#     frame_id: str
#     prompt: str
#     detections: List[Dict] = field(default_factory=list)
#     seq: int = 0


# class LatestDetections:
#     """Same latest-message-wins pattern used in grounding_dino_node: only
#     the newest detections set is ever processed, older ones are dropped."""

#     def __init__(self):
#         self._lock = threading.Lock()
#         self._item: Optional[Detections] = None
#         self._seq = 0
#         self._last_consumed_seq = -1

#     def set(self, item: Detections):
#         with self._lock:
#             self._seq += 1
#             item.seq = self._seq
#             self._item = item

#     def get_if_new(self) -> Optional[Detections]:
#         with self._lock:
#             if self._item is None or self._item.seq == self._last_consumed_seq:
#                 return None
#             self._last_consumed_seq = self._item.seq
#             return self._item


# class SamNode(Node):
#     def __init__(self):
#         super().__init__("sam_node")
#         self.bridge = CvBridge()

#         self.engine = SamEngine()
#         self.get_logger().info("Loading SAM model (this can take a while)...")
#         self.engine.load()
#         self.get_logger().info("Model loaded.")

#         # -- inline frame cache: stamp (sec, nanosec) -> rgb numpy array --
#         self.frame_cache_lock = threading.Lock()
#         self.frame_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

#         self.latest_detections = LatestDetections()

#         rgb_qos = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1,
#         )
#         self.create_subscription(Image, "/zedx1/rgb", self._on_rgb, rgb_qos)
#         self.create_subscription(String, "/dino/detections", self._on_detections, 10)

#         self.mask_pub = self.create_publisher(Image, "/sam/instance_mask", 10)
#         self.instances_pub = self.create_publisher(String, "/sam/instances", 10)

#         self._last_annotated = None
#         self._annotated_lock = threading.Lock()

#         self._stop = threading.Event()
#         self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
#         self._infer_thread.start()

#     # -- ROS callbacks: kept minimal -----------------------------------------
#     def _on_rgb(self, msg: Image):
#         rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
#         key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
#         with self.frame_cache_lock:
#             self.frame_cache[key] = rgb
#             while len(self.frame_cache) > FRAME_CACHE_MAX:
#                 self.frame_cache.popitem(last=False)  # evict oldest

#     def _on_detections(self, msg: String):
#         try:
#             payload = json.loads(msg.data)
#             print("got the file nigg! \n")    #DEBUG
#         except json.JSONDecodeError:
#             self.get_logger().warning("Bad JSON on /dino/detections, skipping.")
#             return

#         header = payload.get("header", {})
#         key = (header.get("stamp_sec"), header.get("stamp_nanosec"))
#         item = Detections(
#             stamp_key=key,
#             frame_id=header.get("frame_id", ""),
#             prompt=payload.get("prompt", ""),
#             detections=payload.get("detections", []),
#         )
#         self.latest_detections.set(item)

#     def _lookup_frame(self, key: tuple) -> Optional[np.ndarray]:
#         with self.frame_cache_lock:
#             return self.frame_cache.get(key)

#     # -- background inference thread -----------------------------------------
#     def _inference_loop(self):
#         while not self._stop.is_set():
#             item = self.latest_detections.get_if_new()
#             if item is None:
#                 time.sleep(0.005)
#                 continue

#             rgb = self._lookup_frame(item.stamp_key)
#             print("Found cached frame:", rgb is not None)
#             if rgb is None:
#                 self.get_logger().debug(
#                     f"No cached frame for stamp {item.stamp_key} - "
#                     f"increase FRAME_CACHE_MAX if this happens often."
#                 )
#                 continue

#             boxes = [d["box_xyxy"] for d in item.detections]
#             labels = [d["label"] for d in item.detections]
#             scores = [d["score"] for d in item.detections]

#             try:
#                 masks = self.engine.infer(rgb, boxes) if boxes else []
#             except Exception as e:
#                 self.get_logger().error(f"SAM inference failed, skipping this frame: {e}")
#                 time.sleep(0.05)
#                 continue
            


#             print(rgb.shape)
#             print(rgb.dtype)
#             print(rgb.min(), rgb.max())

#             annotated, instance_mask = self._build_outputs(rgb, masks, labels, scores)
#             with self._annotated_lock:
#                 self._last_annotated = annotated

#             self._publish(item, instance_mask, labels, scores, boxes)

#     def _build_outputs(self, rgb, masks, labels, scores):
#         overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
#         instance_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

#         rng = np.random.default_rng(42)  # stable colors across frames
#         for i, mask in enumerate(masks):
#             instance_id = i + 1  # 0 reserved for background
#             instance_mask[mask] = instance_id

#             color = rng.integers(0, 255, size=3).tolist()
#             colored = np.zeros_like(overlay)
#             colored[mask] = color
#             overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)

#             ys, xs = np.where(mask)
#             if len(xs) > 0:
#                 cx, cy = int(xs.mean()), int(ys.mean())
#                 cv2.putText(overlay, f"{labels[i]} {scores[i]:.2f}", (cx, cy),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

#         return overlay, instance_mask

#     def _publish(self, item: Detections, instance_mask, labels, scores, boxes):
#         mask_msg = self.bridge.cv2_to_imgmsg(instance_mask, encoding="mono8")
#         mask_msg.header.stamp.sec = item.stamp_key[0] or 0
#         mask_msg.header.stamp.nanosec = item.stamp_key[1] or 0
#         mask_msg.header.frame_id = item.frame_id
#         self.mask_pub.publish(mask_msg)

#         instances = [
#             {"instance_id": i + 1, "label": labels[i], "score": scores[i], "box_xyxy": boxes[i]}
#             for i in range(len(labels))
#         ]
#         payload = {
#             "header": {
#                 "stamp_sec": item.stamp_key[0],
#                 "stamp_nanosec": item.stamp_key[1],
#                 "frame_id": item.frame_id,
#             },
#             "prompt": item.prompt,
#             "instances": instances,
#         }
#         out = String()
#         out.data = json.dumps(payload)
#         self.instances_pub.publish(out)

#     # -- display loop: MUST run on the main thread ---------------------------
#     def display_loop(self):
#         window = "SAM"
#         cv2.namedWindow(window, cv2.WINDOW_NORMAL)
#         while rclpy.ok() and not self._stop.is_set():
#             with self._annotated_lock:
#                 img = self._last_annotated
#             if img is not None:
#                 cv2.imshow(window, img)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 self.get_logger().info("'q' pressed - shutting down.")
#                 self._stop.set()
#                 break

#             rclpy.spin_once(self, timeout_sec=0.0)

#         cv2.destroyAllWindows()

#     def destroy_node(self):
#         self._stop.set()
#         self._infer_thread.join(timeout=1.0)
#         super().destroy_node()


# def main():
#     rclpy.init()
#     node = SamNode()
#     try:
#         node.display_loop()
#     finally:
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == "__main__":
#     main()
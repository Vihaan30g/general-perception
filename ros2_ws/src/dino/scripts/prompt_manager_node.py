"""
prompt_manager_node.py

Persistent node that holds the "active prompt" and republishes it with
TRANSIENT_LOCAL durability (a "latched" topic) on /dino/active_prompt, so
any node that subscribes - even one started AFTER the prompt was set, like
grounding_dino_node - immediately receives the last known prompt instead of
waiting for the next change.

Change the prompt at runtime from another terminal with:
    ros2 run dino set_prompt.py "yellow mug . red bottle . green table ."

Starts with an empty prompt on purpose: until you set one, downstream nodes
should produce zero detections.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String


class PromptManagerNode(Node):
    def __init__(self):
        super().__init__("prompt_manager_node")

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._current_prompt = ""
        self._pub = self.create_publisher(String, "/dino/active_prompt", latched_qos)
        self.create_subscription(String, "/dino/prompt_cmd", self._on_prompt_cmd, 10)

        self.get_logger().info(
            "prompt_manager_node ready. No prompt set yet - downstream nodes "
            "will produce no boxes until you run set_prompt.py."
        )

    def _on_prompt_cmd(self, msg: String):
        self._current_prompt = msg.data.strip()
        self.get_logger().info(f"Active prompt updated: '{self._current_prompt}'")
        out = String()
        out.data = self._current_prompt
        self._pub.publish(out)


def main():
    rclpy.init()
    node = PromptManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

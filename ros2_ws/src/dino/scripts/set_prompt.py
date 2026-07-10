"""
set_prompt.py - one-shot CLI utility, not a persistent node.

Usage:
    ros2 run dino set_prompt.py "yellow mug . red bottle . green table ."

Publishes the given prompt once to /dino/prompt_cmd (which prompt_manager_node
is listening on) and exits. Multiple object names go in ONE string, each
phrase lowercase and ending with " . " - this is the format Grounding DINO
expects.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    if len(sys.argv) < 2:
        print('Usage: ros2 run dino set_prompt.py "yellow mug . red bottle . green table ."')
        sys.exit(1)

    prompt = sys.argv[1]

    rclpy.init()
    node = Node("set_prompt_client")
    pub = node.create_publisher(String, "/dino/prompt_cmd", 10)

    # Brief pause so the publisher has time to discover/match the subscriber
    # before we publish - otherwise a single publish() right after creation
    # can be sent before anyone is listening and simply vanish.
    time.sleep(0.3)

    msg = String()
    msg.data = prompt
    pub.publish(msg)
    node.get_logger().info(f"Sent prompt: '{prompt}'")

    time.sleep(0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

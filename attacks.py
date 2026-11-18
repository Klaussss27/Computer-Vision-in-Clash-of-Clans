"""Validate and execute user-defined attack steps."""

import time


def validate_attack(steps, size):
    if not isinstance(steps, list) or not steps:
        raise ValueError("Add steps to attack_steps before running live")
    width, height = size
    for step in steps:
        if step.get("action") == "wait":
            if not isinstance(step.get("seconds"), (int, float)) or not 0 <= step["seconds"] <= 240:
                raise ValueError("Wait duration must be between 0 and 240 seconds")
            continue
        if step.get("action") != "click":
            raise ValueError("Attack actions must be click or wait")
        point = step.get("point", [])
        if (
            len(point) != 2
            or not all(isinstance(v, int) for v in point)
            or not (0 <= point[0] < width and 0 <= point[1] < height)
        ):
            raise ValueError("Attack click must be inside capture_size")
        count = step.get("count", 1)
        if type(count) is not int or not 1 <= count <= 300:
            raise ValueError("Click count must be between 1 and 300")
        if (
            not isinstance(step.get("interval", 0.3), (int, float))
            or not 0.05 <= step.get("interval", 0.3) <= 10
        ):
            raise ValueError("Click interval must be between 0.05 and 10 seconds")


def run_attack(steps, click, sleep=time.sleep):
    for step in steps:
        if step["action"] == "wait":
            sleep(step["seconds"])
        else:
            for _ in range(step.get("count", 1)):
                click(step["point"])
                sleep(step.get("interval", 0.3))

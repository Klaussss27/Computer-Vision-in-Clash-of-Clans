"""Army requirements shared by planning, calibration, and execution."""

import math


def jumps_enabled(settings):
    return settings.get("use_jump_spells", True) and settings.get("jump_budget", 1) > 0


def required_army(settings):
    return ("sneaky_goblins", "jump_spell") if jumps_enabled(settings) else ("sneaky_goblins",)


def army_profile(profile, settings):
    kinds = set(required_army(settings))
    selected = dict(profile)
    for key in (
        "templates",
        "empty_templates",
        "variant_templates",
        "empty_count_templates",
        "count_offsets",
        "empty_count_offsets",
        "count_thresholds",
    ):
        if key in selected:
            selected[key] = {kind: value for kind, value in selected[key].items() if kind in kinds}
    return selected


def order_targets(targets, settings):
    allowed = [
        target
        for target in targets
        if target.eligible and (target.layers == 0 or jumps_enabled(settings))
    ]

    def priority(target):
        distance = math.dist(target.spawn, target.collector.ground_center)
        redirected = any(
            math.dist(target.spawn, other.collector.ground_center) + 8 < distance
            for other in allowed
            if other is not target
        )
        return (target.layers, redirected, distance, target.collector.center)

    return sorted(allowed, key=priority)

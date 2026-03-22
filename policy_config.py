# -*- coding: utf-8 -*-
"""
Shared policy configuration for inverse opening selection.
"""

POLICY_TARGET_OPENINGS = (20.0, 35.0, 50.0)
POLICY_LOW_MID_THRESHOLD = 3325.246337890625
POLICY_MID_HIGH_THRESHOLD = 5305.74169921875

POLICY_LABEL = "fixed_triplet_data_driven_thresholds_v1"


def select_policy_opening(target_mass: float) -> float:
    if float(target_mass) < POLICY_LOW_MID_THRESHOLD:
        return POLICY_TARGET_OPENINGS[0]
    if float(target_mass) < POLICY_MID_HIGH_THRESHOLD:
        return POLICY_TARGET_OPENINGS[1]
    return POLICY_TARGET_OPENINGS[2]

"""
Retention scoring helpers.
"""

from __future__ import annotations


def clamp_score(score: float, min_value: float, max_value: float) -> float:
    if score < min_value:
        return min_value
    if score > max_value:
        return max_value
    return score


def apply_floor(score: float, floor_score: float) -> float:
    return max(score, floor_score)


def apply_fetch_bump(
    score: float,
    alpha: float,
    bump_clamp_min: float = -2.0,
    bump_clamp_max: float = 1.0,
) -> float:
    clamped = clamp_score(score, bump_clamp_min, bump_clamp_max)
    return score + alpha * (1 - clamped)


def apply_decay_tick(score: float, beta: float, pressure_multiplier: float) -> float:
    return score - beta * pressure_multiplier

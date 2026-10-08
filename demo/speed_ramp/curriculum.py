"""The speed curriculum's decisions, kept apart from Isaac Lab so the unit tests can check them (RECIPE.md).

Runs on: GPU host (imported by train_v3.py); laptop for the unit tests.

- Warm-up: the first `warmup` iterations train the critic only; level 1 (v_max = start_speed) starts after them.
- Isaac gate: a MuJoCo check is due when the level has run `min_iters` iterations, the tracking window is full,
  and over the window frontier robots track >= `track_ratio` of their target with <= `max_fall_rate` of at least
  `min_episodes` finished frontier episodes ending in a fall. Checks are at least `gate_cooldown` apart.
- MuJoCo gate (criteria.py): passing it promotes the level: v_max rises by `step`.
- Stopping rule (note 6): a level that runs `patience` iterations without promotion ends the run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

STAT_KEYS = ("ratio_sum", "ratio_n", "f_falls", "f_done", "a_falls", "a_done")


@dataclass(frozen=True)
class Rules:
    warmup: int
    start_speed: float
    step: float
    min_iters: int
    window: int
    min_episodes: int
    track_ratio: float
    max_fall_rate: float
    gate_cooldown: int
    patience: int

    @classmethod
    def from_args(cls, a) -> Rules:
        return cls(**{k: getattr(a, k) for k in cls.__dataclass_fields__})


@dataclass
class State:
    stage: int  # promoted stages so far (0 = Booster's k1_walk)
    v_max: float  # the level being trained
    level_start: int  # iteration the level started
    last_gate_try: int | None = None  # iteration of the last MuJoCo check

    @classmethod
    def fresh(cls, rules: Rules) -> State:
        return cls(stage=0, v_max=rules.start_speed, level_start=rules.warmup)


def window_stats(window: Iterable[dict]) -> dict:
    """Frontier tracking and fall rates over the window's per-iteration sums (SpeedCurriculumCommand.pop_stats)."""
    s = dict.fromkeys(STAT_KEYS, 0.0)
    for w in window:
        for k in STAT_KEYS:
            s[k] += w[k]
    return {
        "frontier_tracking": s["ratio_sum"] / s["ratio_n"] if s["ratio_n"] else 0.0,
        "frontier_fall_rate": s["f_falls"] / s["f_done"] if s["f_done"] else 1.0,
        "frontier_episodes": int(s["f_done"]),
        "overall_fall_rate": s["a_falls"] / s["a_done"] if s["a_done"] else 1.0,
    }


def gate_due(state: State, rules: Rules, it: int, stats: dict, window_len: int) -> bool:
    """Whether the Isaac gate passes at iteration `it`, so the MuJoCo check runs now."""
    return (
        it - state.level_start >= rules.min_iters
        and window_len >= rules.window
        and stats["frontier_episodes"] >= rules.min_episodes
        and stats["frontier_tracking"] >= rules.track_ratio
        and stats["frontier_fall_rate"] <= rules.max_fall_rate
        and (state.last_gate_try is None or it - state.last_gate_try >= rules.gate_cooldown)
    )


def promote(state: State, rules: Rules, it: int) -> State:
    """The state after the level passes both gates at iteration `it`."""
    return State(stage=state.stage + 1, v_max=round(state.v_max + rules.step, 4), level_start=it, last_gate_try=it)


def out_of_patience(state: State, rules: Rules, it: int) -> bool:
    """Recipe note 6: the level has run `patience` iterations without promotion."""
    return it - state.level_start >= rules.patience


def state_from_events(events: list[dict], rules: Rules, up_to: int) -> State:
    """Rebuild the curriculum from ramp_events.jsonl, ignoring anything after iteration `up_to`.

    Raises ValueError if the run was already stopped by the stopping rule.
    """
    evs = [e for e in events if "iteration" in e and e["iteration"] <= up_to]
    stopped = [e for e in evs if e["event"] == "stopped"]
    if stopped:
        raise ValueError(f"the run was stopped by the stopping rule at iteration {stopped[-1]['iteration']}; "
                         "there is nothing to resume")
    hits = [e for e in evs if e["event"] == "target_hit"]
    tries = [e["iteration"] for e in evs if e["event"] in ("target_hit", "validation_failed", "validation_error")]
    return State(
        stage=hits[-1]["stage"] if hits else 0,
        v_max=round(hits[-1]["target"] + rules.step, 4) if hits else rules.start_speed,
        level_start=hits[-1]["iteration"] if hits else rules.warmup,
        last_gate_try=max(tries) if tries else None,
    )

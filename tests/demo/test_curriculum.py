"""The speed curriculum's decisions (demo/speed_ramp/curriculum.py). Runs on: laptop."""

from types import SimpleNamespace

import pytest
from curriculum import Rules, State, gate_due, out_of_patience, promote, state_from_events, window_stats

RULES = Rules(warmup=100, start_speed=1.0, step=0.25, min_iters=300, window=200, min_episodes=200, track_ratio=0.9,
              max_fall_rate=0.02, gate_cooldown=100, patience=1000)
GOOD = {"frontier_tracking": 0.95, "frontier_fall_rate": 0.01, "frontier_episodes": 500, "overall_fall_rate": 0.01}

# Shape of a real run's ramp_events.jsonl (abridged).
EVENTS = [
    {"event": "start", "start_speed": 1.0, "step": 0.25},
    {"event": "actor_unfrozen", "iteration": 99},
    {"event": "validation_failed", "target": 1.0, "iteration": 1000},
    {"event": "target_hit", "stage": 1, "target": 1.0, "iteration": 1300},
    {"event": "target_hit", "stage": 2, "target": 1.25, "iteration": 1600},
    {"event": "validation_failed", "target": 1.5, "iteration": 1800},
    {"event": "target_hit", "stage": 3, "target": 1.5, "iteration": 1900},
    {"event": "validation_error", "target": 1.75, "iteration": 2300},
]


def test_rules_from_args_takes_the_named_fields():
    a = SimpleNamespace(**vars(RULES) | {"seed": 3, "rec_python": "/x"})
    assert Rules.from_args(a) == RULES


def test_fresh_state_starts_after_warmup():
    assert State.fresh(RULES) == State(stage=0, v_max=1.0, level_start=100, last_gate_try=None)


def test_window_stats():
    win = [{"ratio_sum": 9.0, "ratio_n": 10, "f_falls": 1, "f_done": 50, "a_falls": 2, "a_done": 100}] * 4
    st = window_stats(win)
    assert st == {"frontier_tracking": 0.9, "frontier_fall_rate": 0.02, "frontier_episodes": 200, "overall_fall_rate": 0.02}
    empty = window_stats([])
    assert empty["frontier_tracking"] == 0.0 and empty["frontier_fall_rate"] == 1.0  # an empty window never passes


def test_gate_waits_for_min_iters_and_a_full_window():
    s = State.fresh(RULES)
    assert not gate_due(s, RULES, 399, GOOD, 200)  # 299 iterations on the level
    assert not gate_due(s, RULES, 400, GOOD, 199)
    assert gate_due(s, RULES, 400, GOOD, 200)


@pytest.mark.parametrize("change", [{"frontier_tracking": 0.89}, {"frontier_fall_rate": 0.021},
                                    {"frontier_episodes": 199}])
def test_gate_needs_tracking_low_falls_and_enough_episodes(change):
    assert not gate_due(State.fresh(RULES), RULES, 400, GOOD | change, 200)


def test_gate_cooldown_between_mujoco_checks():
    s = State(stage=0, v_max=1.0, level_start=100, last_gate_try=400)
    assert not gate_due(s, RULES, 499, GOOD, 200)
    assert gate_due(s, RULES, 500, GOOD, 200)


def test_promotion_raises_v_max_and_starts_a_new_level():
    s = promote(State(stage=2, v_max=1.5, level_start=1600, last_gate_try=1800), RULES, 1900)
    assert s == State(stage=3, v_max=1.75, level_start=1900, last_gate_try=1900)
    assert not gate_due(s, RULES, 2000, GOOD, 200)  # a new level needs its own min_iters


def test_stopping_rule_counts_from_the_level_start():
    s = State(stage=5, v_max=2.25, level_start=2500, last_gate_try=3400)
    assert not out_of_patience(s, RULES, 3499)
    assert out_of_patience(s, RULES, 3500)


def test_state_after_last_promotion():
    s = state_from_events(EVENTS, RULES, up_to=1950)
    assert s == State(stage=3, v_max=1.75, level_start=1900, last_gate_try=1900)


def test_events_after_the_checkpoint_are_ignored():
    # Resuming from iteration 1750: the 1800 failure and the 1900 promotion are redone.
    s = state_from_events(EVENTS, RULES, up_to=1750)
    assert s == State(stage=2, v_max=1.5, level_start=1600, last_gate_try=1600)


def test_broken_checks_count_as_gate_attempts():
    assert state_from_events(EVENTS, RULES, up_to=2400).last_gate_try == 2300


def test_fresh_run_state():
    assert state_from_events(EVENTS[:2], RULES, up_to=150) == State.fresh(RULES)


def test_a_stopped_run_cannot_be_resumed():
    events = [*EVENTS, {"event": "stopped", "iteration": 2900}]
    assert state_from_events(events, RULES, up_to=2850).stage == 3  # checkpoint before the stop: resumable
    with pytest.raises(ValueError, match="stopped by the stopping rule at iteration 2900"):
        state_from_events(events, RULES, up_to=2900)

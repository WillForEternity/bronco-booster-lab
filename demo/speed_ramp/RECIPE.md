# The version-3 fine-tuning recipe

This is the recipe `train_v3.py` implements: it fine-tunes Booster's `k1_walk` for speed in Isaac Lab, and promotes a new speed stage only once the policy also passes Booster's MuJoCo player. Code comments refer to the numbered notes below as "recipe note N".

Every result is simulation only. Nothing here has run on a physical robot.

## The recipe

- **Start:** Booster's `k1_walk` actor from `booster_deploy` @ `7bb1462`. It is exported without a critic, so a new critic is trained for 100 iterations with the actor frozen.
- **Task:** `k1_speed_env.make_env_cfg` (`Bronco-K1-SpeedRamp-v0`). The K1 conventions it shares with export and playback live in `k1_conventions.py`.
  - Inputs, outputs, gains and torque limits copy `booster_deploy`'s `k1_walk` player, with Booster's K1 motor model (torque-speed curve and command delay).
  - Flat ground, 4,096 robots, 20 s episodes.
  - An episode ends if the trunk touches the ground, tilts more than 57°, or drops below 0.3 m.
- **Curriculum:** commands are drawn from [0, v_max], half of them at the frontier v_max, resampled every 10 s. v_max starts at 1.0 m/s and rises by 0.25 m/s per stage.
- **Isaac gate:** at least 300 iterations at the level, then over 200 iterations at least 90% frontier tracking and at most 2% frontier falls.
- **MuJoCo gate:** after the Isaac gate passes, the candidate runs 20 perturbed trials (seeds 0–19) in `booster_deploy`'s MuJoCo player at v_max. It is promoted only if at least 18 trials *succeed*: survive 10 s with their own forward speed at ≥ 90% of v_max (note 3). This is the final test's rule (`criteria.py`), applied to different seeds. Retries wait 100 iterations.
- **Stopping rule:** a level that runs 1,000 iterations without promotion ends the run (note 6).
- **Reward:**
  - velocity tracking (xy in the yaw frame, std 0.5) and heading;
  - termination −200;
  - mechanical power −7e-4 (Fu et al., CoRL 2021: tracking plus energy alone produces natural gaits);
  - foot air time +0.5 (threshold 0.4 s), foot slide −0.1;
  - joint limits −1, hip roll/yaw deviation −0.1, arm deviation −0.1 (note 1), flat orientation −1, xy angular velocity −0.05, action rate −0.005, hip and knee acceleration −1.25e-7.
- **Symmetry:** mirrored-data augmentation in PPO (`k1_symmetry.py`; Mittal et al., ICRA 2024). Checked on `k1_walk` in MuJoCo: 1.9% relative equivariance error with the correct mirror, against 46% and 181% for wrong sign conventions.
- **Randomization:** motor command delay 0–40 ms, friction 0.4–1.2, motor stiffness and damping ×0.8–1.2, trunk mass ±1 kg, pushes ±0.4 m/s per axis every 5–10 s.
- **Arm PD:** simulated implicitly in Isaac, with lower damping on two arm joints (note 2). The same damping is used in MuJoCo with `--damping_profile v3`.
- **Final test** (`final_test.py`, run once after training stops and never used for training decisions): MuJoCo trial seeds 1000–1019 at each promoted stage's v_max (notes 3–7).

## Notes

1. **Arm-deviation penalty.** Without it, the first candidate spent 251 W in its arms in MuJoCo (`k1_walk`: 4 W) and missed the speed bar. It is kept at −0.1, Isaac Lab's G1 value; moderate arm swing remains.
2. **Arm PD chatter.** Explicit PD is stable only while damping × dt / inertia < 2. The elbow-pitch inertia falls to 0.00148 kg m² in some postures, so Booster's damping of 2.0 at MuJoCo's 2 ms step gives 2.69: a 250 Hz chatter at the ±14 Nm torque limit (124 W per elbow). Shoulder pitch reaches 1.94. `k1_walk` keeps its elbows where the ratio is 1.06, but fine-tuned policies move into the unstable range. So:
   - in Isaac, arm and head PD are simulated implicitly;
   - damping is lowered only on the two unstable joints, elbow pitch 2.0 → 0.7 and shoulder pitch 2.0 → 1.0, in both simulators (`--damping_profile v3`);
   - the energy weight was re-measured and set to −7e-4.

   With this, Isaac and MuJoCo agree on `k1_walk` at 1.0 m/s (1.08 against 1.078 m/s, 131 W against about 135 W). These two damping values differ from Booster's `k1_walk` deploy config and need Booster's review before any hardware use.
3. **Per-trial success.** The mean speed of the surviving trials can hide a two-mode outcome (most trials at speed, a few standing still). A final-test trial succeeds only if it survives 10 s *and* its own speed is at least 90% of the command. A stage passes with at least 18 of 20 successes.
4. **Overshoot, and capping commands.** The final test reports each stage's median speed error, (speed − command) / command, and flags a stage above +10% as "overshoots". Commands above a policy's trained v_max are outside its training range and usually fall, so players should cap the command at the stage's v_max.
5. **EPTE-SP.** The final test also reports EPTE-SP (episodic percentage tracking error with stability penalty; Li, Li and Hutter, RA-L 2026, arXiv:2601.17428, Eq. 8): the percentage speed-tracking error averaged over a trial, with every step after a fall counted as 100%.
6. **Stopping rule.** A seed that spends 1,000 iterations on one level without promotion has reached the recipe's ceiling and is stopped (`train_v3.py --patience`, counted from the level's start; the first level starts after the warm-up). For each seed, the result is the highest promoted stage that passes the final test. With several seeds, the recipe's result is the lowest seed's; a higher seed is reported, not claimed.
7. **How the final test measures.** Per-trial speed is forward speed along the trunk's heading (`--speed_metric steady` uses straight-line displacement instead). Energy (cost of transport) and leg torque saturation are measured at every 2 ms physics substep. "Symmetric" means the survivors' mean stance asymmetry is at most 10%. Flight fraction is reported; a flight phase is what separates running from walking.

## Results of the October 7–8 runs

Two seeds, stopped by the stopping rule at v_max 2.25 m/s, where every MuJoCo gate check failed although Isaac tracking was 97–98%. Final test, on seeds 1000–1019:

| Stage | v_max | Seed 1 | Seed 2 |
|---|---|---|---|
| 1 | 1.00 m/s | 12/20, fails | 11/20, fails |
| 2 | 1.25 m/s | 20/20 | 12/20, fails |
| 3 | 1.50 m/s | 20/20 | 20/20 |
| 4 | 1.75 m/s | 20/20 | 20/20 |
| 5 | 2.00 m/s | **20/20** | **19/20** |

The recipe's result is 2.0 m/s. At that stage both feet are off the ground about 26% of the time (a run), and neither seed meets the 10% stance-asymmetry bar (13.4% and 10.1%). All trials of the failed stages survived, but ran about 9% slow.

## Changes since the October 7–8 runs

The code here differs from what produced the results above in two ways that change training, listed below. The other changes are to the tools around it (the parity check now checks the training task, `--resume` keeps a run's own settings, Booster's stage-0 `k1_walk` is recorded with Booster's damping) and to the code's structure. The restructuring was checked against the old code: the same policies give identical MuJoCo trials, identical exports and identical final-test scores, and the Isaac task's configuration is unchanged.

1. **The MuJoCo gate uses the final test's per-trial rule.** It used to promote a stage when 18 trials survived and the survivors' *mean* straight-line speed reached 90% of v_max. Three of the stages above (stage 1 of both seeds, stage 2 of seed 2) passed that gate by less than 0.01 m/s and then failed the final test. With one rule, a promoted stage is one the final test can be expected to pass, and training stops spending time on stages that would fail.
2. **The stopping rule is part of the trainer.** The October runs were stopped by a separate watcher process; `train_v3.py` used to log a "stalled" event after 3,000 iterations and keep training.

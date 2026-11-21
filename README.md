# OBELIX: Autonomous Warehouse Robot (CS780 Capstone)

**Course:** CS780 — Deep Reinforcement Learning (IIT Kanpur)  
**Advisor:** Prof. Ashutosh Modi  
**Author:** Khush Maheshwari (`khushmahes23@iitk.ac.in`)

Training an autonomous warehouse robot from scratch in PyTorch to locate, attach to, and push a box to the arena boundary under partial observability (POMDP), sparse rewards, and aliased wall obstacles.

---

## Demo Video

[![OBELIX Demo Video](https://img.youtube.com/vi/8aFcyrNMf4U/hqdefault.jpg)](https://www.youtube.com/watch?v=8aFcyrNMf4U&t=77s)

Watch the agent in action: [YouTube Demo (starts at 1:17)](https://www.youtube.com/watch?v=8aFcyrNMf4U&t=77s)

---

## Problem Formulation

The environment recreates a harder POMDP version of Mahadevan & Connell's (1991) box-pushing task:

- **Observation Space (18 binary bits):**
  - `16` sonar bits (`8` near-zone + `8` far-zone across 8 directions, range `~0–150 px`)
  - `1` infrared (IR) bit (range `~20 px`, fires only when the box is directly ahead)
  - `1` stuck flag (fires when forward movement is blocked)
- **Action Space (5 discrete actions):** `L45`, `L22`, `FW`, `R22`, `R45`
- **Sparse Reward Structure:** `-1` per step with no sensor hits, `-200` per stuck step, `+100` for box attachment, `+2000` for pushing the box to the arena boundary.
- **Three Difficulty Tiers (with optional vertical wall obstacles):**
  - **Phase 1 (Level 1):** Static box
  - **Phase 2 (Level 2):** Blinking box (robot cannot attach while invisible)
  - **Phase 3 (Level 3):** Moving + blinking box

---

## Core Challenge & Key Architecture

### Diagnosing Wall–Box Sensor Aliasing
The simulator merges wall and box pixels before computing sonar hits, making `"wall straight ahead"` and `"box straight ahead"` indistinguishable in a single frame. Policy-gradient baselines (`VPG`, `PPO`) and naive `DDQN` repeatedly collided into walls (`-49,651` mean reward).

### Key Solutions
1. **Wall-Aware Dueling DDQN + PER (`PHASE 1`):**
   - Replaced on-policy rollouts with **Double DQN**, **Dueling value/advantage heads**, and **Prioritized Experience Replay (PER)** to amplify rare `+100` / `+2000` transitions.
   - Added `k=4` frame stacking and temporal disambiguation features (**`recent_stuck_count`** over 10 steps + **`ir_streak`**) to separate persistent wall collisions from true box detections (`80-dim` state), paired with targeted reward shaping (escape bonus, rotation-loop penalty, wall-confusion penalty).
2. **Blinking-Target Memory (`PHASE 2`):**
   - Explored **FeatureBuilder** (`23-dim` episodic memory), **DRQN**, and **CHIMERA** (`k=32` frame stack, `576-dim`, `1M` replay buffer) with two-timescale loop detection and wall-crossing subgoal bonuses.
3. **Moving + Blinking Interception (`PHASE 3`):**
   - Added **dead reckoning** (`x, y, θ` action integration), curriculum learning, ghost-box velocity tracking, and **Q-value action masking** ($Q[\text{FW}] = -\infty$ when stuck after `FW`).

---

## Benchmark Results (10 Evaluation Seeds)

| Phase | Model | No-Wall Reward | Wall-Obstacle Reward |
| :--- | :--- | :---: | :---: |
| **Phase 1 (Static)** | VPG / REINFORCE | `-20,000` | `-48,900` |
| **Phase 1 (Static)** | PPO | `-29,287` | `-49,651` |
| **Phase 1 (Static)** | Basic DDQN | `+934` | `-38,931` |
| **Phase 1 (Static)** | Dueling DDQN + PER | `-3,406` | `-4,174` |
| **Phase 1 (Static)** | 78-dim + Frame Stack | `-2,224` | `-1,975` |
| **Phase 1 (Static)** | **80-dim Wall-Aware Dueling DDQN (Final)** | **`-1,765`** | **`-1,294`** |
| **Phase 2 (Blinking)** | FeatureBuilder (23-dim) | **`-1,279`** | `-4,486` |
| **Phase 2 (Blinking)** | **CHIMERA (576-dim, 1M Replay)** | `-3,806` | **`-2,195`** |

- **97.4% reduction** in Level 1 wall-obstacle penalty (`-49,651` PPO $\rightarrow$ `-1,294`)
- **51% reduction** in Level 2 wall-obstacle penalty (`-4,486` $\rightarrow$ `-2,195`)
- **Codabench Standing:** **Rank 11** on Level 1 (static)

---

## Repository Structure

```text
├── PHASE 1/
│   ├── agent_vpg.py / agent_vpgcorr.py   # Policy gradient baselines (VPG / REINFORCE)
│   ├── agent_ppo.py                      # PPO Actor-Critic baseline
│   ├── agent_ddqn.py / train_ddqn.py     # Double DQN + uniform replay
│   ├── agent_ddqnper.py / train_ddqnper.py
│   ├── agent_ddqn_onlywallbetter.py      # Final 80-dim wall-aware Dueling DDQN + PER
│   └── RAINBOW/                          # Rainbow C51 + NoisyNets + n-step experiments
├── PHASE 2/
│   ├── og_agent.py / og_train.py         # CHIMERA (32-frame stack, 1M replay)
│   ├── agent_drqn.py / train_drqn.py     # Recurrent DRQN experiments
│   └── agent_phase2_v2.py                # FeatureBuilder episodic memory agent
└── PHASE 3/
    ├── agent_2.py / agent_3.py           # Dead reckoning + curriculum + action masking
    └── final.py                          # Final integrated evaluation agent
```

---

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `Deep Reinforcement Learning (Dueling DDQN, PER, PPO, Rainbow C51, DRQN)`

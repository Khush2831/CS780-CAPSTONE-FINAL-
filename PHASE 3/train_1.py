"""
CHIMERA TRAINER v4.1: ULTIMATE REFINEMENT
"""

from __future__ import annotations
import argparse
import random
import os
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# [CPU OPTIMIZATION]: Force PyTorch to use all available physical CPU cores
torch.set_num_threads(os.cpu_count() if os.cpu_count() else 4)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
FRAME_STACK = 32  
OBS_DIM = 18
STATE_DIM = OBS_DIM * FRAME_STACK

# ---------------------------------------------------------
# 0. HYPER-OPTIMIZED FRAME STACK WRAPPER
# ---------------------------------------------------------
class FrameStackWrapper:
    def __init__(self, env, k=32):
        self.env = env
        self.k = k
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        
    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.state[:] = np.tile(obs, self.k)
        return self.state
        
    def step(self, action, render=False):
        obs, reward, done = self.env.step(action, render=render)
        self.state[:-OBS_DIM] = self.state[OBS_DIM:]
        self.state[-OBS_DIM:] = obs
        return self.state, reward, done

# ---------------------------------------------------------
# 1. DUELING NETWORK ARCHITECTURE
# ---------------------------------------------------------
class DuelingDQN(nn.Module):
    def __init__(self, in_dim=STATE_DIM, n_actions=5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

# ---------------------------------------------------------
# 2. CPU-VECTORIZED SUMTREE PER 
# ---------------------------------------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0
        self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, p):
        idx = self.write + self.capacity - 1
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_batch(self, s_array):
        idx = np.zeros(len(s_array), dtype=np.int32)
        while True:
            active = idx < (self.capacity - 1)
            if not np.any(active):
                break

            left = 2 * idx + 1
            right = left + 1
            
            safe_left = np.where(active, left, 0)
            left_vals = self.tree[safe_left]
            
            mask = s_array > left_vals
            
            s_array = np.where(active & mask, s_array - left_vals, s_array)
            idx = np.where(active, np.where(mask, right, left), idx)
            
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx

    @property
    def total_p(self):
        return self.tree[0]

class PrioritizedReplay:
    def __init__(self, cap: int = 1_000_000, alpha: float = 0.6):
        self.cap = cap
        self.alpha = alpha
        self.tree = SumTree(cap)
        self.max_priority = 1.0
        
        self.s_buf = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.a_buf = np.zeros(cap, dtype=np.int64)
        self.r_buf = np.zeros(cap, dtype=np.float32)
        self.s2_buf = np.zeros((cap, STATE_DIM), dtype=np.float32)
        self.d_buf = np.zeros(cap, dtype=np.float32)

    def add(self, s, a, r, s2, d):
        tree_idx = self.tree.write + self.tree.capacity - 1
        idx = self.tree.write
        self.s_buf[idx] = s
        self.a_buf[idx] = a
        self.r_buf[idx] = r
        self.s2_buf[idx] = s2
        self.d_buf[idx] = d
        
        p = self.max_priority ** self.alpha
        self.tree.add(p)
        return tree_idx

    def sample(self, batch: int, beta: float = 0.4):
        segment = self.tree.total_p / batch
        points = np.random.uniform(
            segment * np.arange(batch), 
            segment * (np.arange(batch) + 1)
        )

        batch_idx, priorities, data_idxs = self.tree.get_batch(points)

        probs = priorities / self.tree.total_p
        total = self.tree.n_entries
        
        weights = (total * probs) ** (-beta)
        weights /= weights.max()
        batch_weights = weights.astype(np.float32)

        s = self.s_buf[data_idxs]
        act = self.a_buf[data_idxs]
        r = self.r_buf[data_idxs]
        s2 = self.s2_buf[data_idxs]
        d = self.d_buf[data_idxs]

        return s, act, r, s2, d, batch_idx, batch_weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.max_priority = max(self.max_priority, prio)
            self.tree.update(idx, prio ** self.alpha)

    def __len__(self): return self.tree.n_entries

# ---------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="obelix.py")
    ap.add_argument("--out", type=str, default="weights_lvl1.pth") 
    
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=2000) 
    
    ap.add_argument("--difficulty", type=int, default=0) 
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--resume", action="store_true", help="Load existing weights")
    ap.add_argument("--resume_steps", type=int, default=0, help="Steps already taken")

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--batch", type=int, default=512) 
    
    ap.add_argument("--replay", type=int, default=1_000_000) 
    ap.add_argument("--warmup", type=int, default=20000) 
    ap.add_argument("--target_sync", type=int, default=4000) 
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.02) 
    ap.add_argument("--eps_decay_episodes", type=int, default=1500) 
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    print(f"🔥 TRAINING LEVEL 1 (STATIC BOX) | Optimized for: {device} ({os.cpu_count()} threads)")

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingDQN().to(device)
    tgt = DuelingDQN().to(device)

    steps = 0
    best_avg_score = -float('inf')

    if args.resume and os.path.exists(args.out):
        print(f"Loading existing weights from {args.out}...")
        q.load_state_dict(torch.load(args.out, map_location=device, weights_only=True))
        
        best_file_name = args.out.replace(".pth", "_best.pth")
        if os.path.exists(best_file_name):
            print("Preserving historical high score to prevent overwrites...")
            
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = PrioritizedReplay(args.replay, alpha=0.6)

    def eps_by_episode(current_ep):
        if current_ep >= args.eps_decay_episodes:
            return args.eps_end
        frac = current_ep / float(args.eps_decay_episodes)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    cumulative_total_reward = 0.0
    recent_returns = deque(maxlen=50) 
    recent_successes = deque(maxlen=50) 

    TRAIN_FREQ = 4

    def get_target_coords(bx, by, tx, ty, walls_on):
        if not walls_on:
            return tx, ty

        # Door is strictly at X=240 to 260. Y=250.
        if tx > 260 and bx < 240:
            if bx < 210:
                return 200.0, 250.0  # Force to Staging Area Left
            return 300.0, 250.0      # Pull through the gap
            
        elif tx < 240 and bx > 260:
            if bx > 290:
                return 300.0, 250.0  # Force to Staging Area Right
            return 200.0, 250.0  
            
        return tx, ty 

    for ep in range(args.episodes):
        
        # 80% Wall ON / 20% Wall OFF Split
        use_walls_this_episode = np.random.rand() < 0.80
        current_difficulty = 0  # Always 0 (Static Box)

        is_eval = (ep > 0 and ep % 20 == 0)
        current_eps = 0.05 if is_eval else eps_by_episode(ep)

        base_env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=use_walls_this_episode,
            difficulty=current_difficulty,
            box_speed=args.box_speed,
            seed=args.seed,
        )
        env = FrameStackWrapper(base_env, k=FRAME_STACK)
        
        s = env.reset(seed=args.seed)
        ep_ret = 0.0
        ep_steps = 0
        
        bot_x, bot_y = base_env.bot_center_x, base_env.bot_center_y
        box_x, box_y = base_env.box_center_x, base_env.box_center_y
        
        start_bot_x = bot_x 
        
        started_opposite = ((start_bot_x < 250 and box_x > 250) or (start_bot_x > 250 and box_x < 250))
        crossed_the_wall = False 
        
        steps_to_cross = -1 
        stuck_frames = 0
        steps_near_wall = 0
        claimed_crossing_reward = False 
        
        target_x, target_y = get_target_coords(bot_x, bot_y, box_x, box_y, use_walls_this_episode)
        prev_dist = math.hypot(bot_x - target_x, bot_y - target_y)
        
        prev_bound_dist = None 
        spin_counter = 0
        
        hist_micro = np.zeros((32, 2), dtype=np.float32)
        hist_macro = np.zeros((120, 2), dtype=np.float32)
        micro_cnt = 0
        macro_cnt = 0

        current_ep_memory_addresses = []

        for _ in range(args.max_steps):
            
            if np.random.rand() < current_eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
                    a = q(s_t).argmax(dim=1).item()

            action_str = ACTIONS[a]
            s2, r, done = env.step(action_str, render=True)
            custom_r = float(r)
            
            bot_x, bot_y = base_env.bot_center_x, base_env.bot_center_y
            box_x, box_y = base_env.box_center_x, base_env.box_center_y
            
            if use_walls_this_episode and 220 < bot_x < 280:
                steps_near_wall += 1
            
            # ---------------------------------------------------------
            # THE FLAWLESS CROSSING LOGIC (Proper Margins)
            # ---------------------------------------------------------
            if use_walls_this_episode and started_opposite and not crossed_the_wall:
                # If starting left, must fully clear right edge (260 + 30 radius = 290)
                if start_bot_x < 250 and bot_x > 290:
                    crossed_the_wall = True
                    steps_to_cross = ep_steps 
                # If starting right, must fully clear left edge (240 - 30 radius = 210)
                elif start_bot_x > 250 and bot_x < 210:
                    crossed_the_wall = True
                    steps_to_cross = ep_steps 
                    
            if crossed_the_wall and not claimed_crossing_reward:
                custom_r += 1000.0  # Upgraded to +1000
                claimed_crossing_reward = True
            
            old_target = (target_x, target_y)
            target_x, target_y = get_target_coords(bot_x, bot_y, box_x, box_y, use_walls_this_episode)
            
            if (target_x, target_y) != old_target:
                prev_dist = math.hypot(bot_x - target_x, bot_y - target_y)

            current_dist = math.hypot(bot_x - target_x, bot_y - target_y)
            sensors_clear = not np.any(base_env.sensor_feedback[:17])
            
            hist_micro[micro_cnt % 32] = [bot_x, bot_y]
            hist_macro[macro_cnt % 120] = [bot_x, bot_y]
            micro_cnt += 1
            macro_cnt += 1
            
            if micro_cnt >= 32:
                spreads_mi = np.ptp(hist_micro, axis=0)
                if spreads_mi[0] < 60.0 and spreads_mi[1] < 60.0:
                    custom_r -= 10.0
                micro_cnt = 0  

            if macro_cnt >= 120:
                spreads_ma = np.ptp(hist_macro, axis=0)
                is_near_door = (225 < bot_x < 275) and (190 < bot_y < 310)
                
                if spreads_ma[0] < 150.0 and spreads_ma[1] < 150.0:
                    if not (use_walls_this_episode and is_near_door):
                        custom_r -= 25.0
                    macro_cnt = 0 

            if base_env.stuck_flag == 1:
                stuck_frames += 1
                if action_str != "FW":
                    custom_r += 150.0  

            if base_env.enable_push:
                dist_x = min(box_x - 10, base_env.frame_size[1] - 10 - box_x)
                dist_y = min(box_y - 10, base_env.frame_size[0] - 10 - box_y)
                current_bound_dist = min(dist_x, dist_y)
                
                if prev_bound_dist is None:
                    prev_bound_dist = current_bound_dist
                    
                progress = prev_bound_dist - current_bound_dist
                custom_r += progress * 2.0  
                prev_bound_dist = current_bound_dist
                
            else:
                if sensors_clear:
                    progress = prev_dist - current_dist
                    custom_r += progress * 1.0  
                    
                    if action_str != "FW":
                        spin_counter += 1
                        if spin_counter > 6:
                            custom_r -= 15.0 
                    else:
                        spin_counter = 0
                else:
                    spin_counter = 0 
                
                prev_dist = current_dist 

            ep_ret += custom_r
            
            if not is_eval:
                mem_addr = replay.add(s, a, custom_r, s2.copy(), float(done))
                current_ep_memory_addresses.append(mem_addr)
                
                steps += 1
                
                if steps % TRAIN_FREQ == 0 and len(replay) >= max(args.warmup, args.batch):
                    beta = min(1.0, 0.4 + steps * (1.0 - 0.4) / 3_000_000)
                    sb, ab, rb, s2b, db, idx_b, weights_b = replay.sample(args.batch, beta=beta)
                    
                    sb_t = torch.as_tensor(sb, dtype=torch.float32)
                    ab_t = torch.as_tensor(ab, dtype=torch.int64)
                    rb_t = torch.as_tensor(rb, dtype=torch.float32)
                    s2b_t = torch.as_tensor(s2b, dtype=torch.float32)
                    db_t = torch.as_tensor(db, dtype=torch.float32)
                    weights_t = torch.as_tensor(weights_b, dtype=torch.float32)

                    with torch.no_grad():
                        next_q = q(s2b_t)
                        next_a = torch.argmax(next_q, dim=1)
                        next_q_tgt = tgt(s2b_t)
                        next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                        y = rb_t + args.gamma * (1.0 - db_t) * next_val

                    pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                    
                    td_errors = torch.abs(pred - y).detach().numpy()
                    replay.update_priorities(idx_b, td_errors + 1e-5)

                    loss = (weights_t * nn.functional.smooth_l1_loss(pred, y, reduction="none")).mean()

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                    opt.step()

                    if steps % args.target_sync == 0:
                        tgt.load_state_dict(q.state_dict())
            
            s = s2.copy()
            ep_steps += 1
            if done:
                break

        # =====================================================
        # SURGICAL TRAJECTORY PRIORITIZATION 
        # =====================================================
        is_success = 1 if ep_steps < args.max_steps else 0
        stuck_percent = (stuck_frames / max(1, ep_steps)) * 100.0

        if use_walls_this_episode:
            if started_opposite:
                cross_str = f"Crossed: YES (Step {steps_to_cross})" if crossed_the_wall else "Crossed: NO"
            else:
                cross_str = "Crossed: N/A (Same Side)"
        else:
            cross_str = "Wall: OFF"

        lvl_str = f"Lvl {current_difficulty} | Wall {'ON' if use_walls_this_episode else 'OFF'}"

        if is_eval:
            print(f"   >>> 🧪 [EVALUATION RUN] [{lvl_str}] Ret: {ep_ret:.1f} | Steps: {ep_steps} | Stuck: {stuck_percent:.1f}% | {cross_str} <<<")
        else:
            recent_successes.append(is_success)
            multiplier = 1.0
            tier_name = ""

            if use_walls_this_episode:
                if started_opposite and crossed_the_wall and is_success == 1:
                    # [UPGRADED]: Master Tier bumped to 200x max priority!
                    multiplier = max(20.0, min(200.0, 200.0 * (1000.0 / max(1, ep_steps))))
                    tier_name = f"TRUE CROSSING MASTER TIER | {ep_steps} steps"
                elif not started_opposite and is_success == 1:
                    multiplier = 40.0
                    tier_name = f"Lucky Spawn Box Push | {ep_steps} steps"
                elif started_opposite and crossed_the_wall and is_success == 0:
                    # [UPGRADED]: Pioneer Tier bumped from 20x to 50x! 
                    # Even if it fails to push the box, we DEMAND it remembers how to cross the door.
                    multiplier = 50.0
                    tier_name = f"CROSSING ACHIEVED BUT FAILED TO PUSH | {ep_steps} steps"
            else:
                if is_success == 1:
                    multiplier = max(5.0, min(25.0, 25.0 * (400.0 / max(1, ep_steps))))
                    tier_name = f"Speedrun (Wall OFF) | {ep_steps} steps"

            if multiplier > 1.0:
                speed_mod = max(0.5, min(1.0, 1000.0 / max(1, ep_steps)))
                final_multiplier = multiplier * speed_mod
                print(f"   --> 🧠 GOLDEN TRAJECTORY! [{tier_name}] Bootstrapping with {final_multiplier:.1f}x Priority!")
                boosted_p = (replay.max_priority * final_multiplier) ** replay.alpha
                for addr in current_ep_memory_addresses:
                    replay.tree.update(addr, boosted_p)

            cumulative_total_reward += ep_ret
            recent_returns.append(ep_ret)
            avg_50 = np.mean(recent_returns)
            win_rate_50 = (sum(recent_successes) / len(recent_successes)) * 100.0

            if (ep + 1) % 1 == 0:  
                print(f"Ep {ep+1}/{args.episodes} [{lvl_str}] | Ret: {ep_ret:.1f} | Avg: {avg_50:.1f} | WinRate: {win_rate_50:.1f}% | eps: {current_eps:.3f} | TotalSteps: {ep_steps} | WallTime: {steps_near_wall} | Stuck: {stuck_percent:.1f}% | {cross_str}")

            if ep >= 50 and avg_50 > best_avg_score:
                best_avg_score = avg_50
                best_file_name = args.out.replace(".pth", "_best.pth")
                torch.save(q.state_dict(), best_file_name)
                print(f"   --> 🌟 NEW HIGH SCORE: {best_avg_score:.1f}! Saved to {best_file_name}")

            if (ep + 1) % 50 == 0:
                torch.save(q.state_dict(), args.out)
                print(f"   --> 💾 Saved regular checkpoint to {args.out}")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------
# EVALUATION POLICY
# ---------------------------------------------------------
_MODEL = None
_EVAL_STATE = None 
_LAST_OBS = None 

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _MODEL, _EVAL_STATE, _LAST_OBS
    
    if _MODEL is None:
        _MODEL = DuelingDQN()
        submission_dir = os.path.dirname(__file__)
        wpath = os.path.join(submission_dir, "weights_lvl1_best.pth")
        
        if os.path.exists(wpath):
            _MODEL.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        else:
            print(f"WARNING: Could not find {wpath}. Trying fallback...")
            fallback_wpath = os.path.join(submission_dir, "weights_lvl1.pth")
            if os.path.exists(fallback_wpath):
                _MODEL.load_state_dict(torch.load(fallback_wpath, map_location="cpu", weights_only=True))
            else:
                print("WARNING: No weights found at all.")
            
        _MODEL.eval()

    if _LAST_OBS is not None:
        sensor_jump = np.linalg.norm(obs[:16] - _LAST_OBS[:16])
        if sensor_jump > 2.0:  
            _EVAL_STATE = None 
            
    _LAST_OBS = obs.copy()

    if _EVAL_STATE is None:
        _EVAL_STATE = np.tile(obs, FRAME_STACK).astype(np.float32)
    else:
        _EVAL_STATE[:-OBS_DIM] = _EVAL_STATE[OBS_DIM:]
        _EVAL_STATE[-OBS_DIM:] = obs
        
    x = torch.from_numpy(_EVAL_STATE).unsqueeze(0)
    with torch.no_grad():
        qs = _MODEL(x).squeeze(0).numpy()
    return ACTIONS[int(np.argmax(qs))]
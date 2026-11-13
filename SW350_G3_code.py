# SW350 Group 3: Pilgrim disorientation simulation and machine learning experiment
# All data are synthetic. Code generated with the assistance of Claude (Anthropic) and executed by the team in Google Colab.
# Run in Google Colab (or Python 3 with numpy, pandas, scikit-learn, matplotlib): python SW350_G3_code.py

# ======================================================================
# Cell 1: Load the libraries
# ======================================================================
import json, os, shutil, sys, time, zipfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "pilgrim_outputs"
os.makedirs(OUT, exist_ok=True)
START = time.time()
VERSIONS = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
print(VERSIONS)

# ======================================================================
# Cell 2: Simulation settings
# ======================================================================
CONFIG = {
    "master_seed": 350,
    "n_pilgrims": 1000, "duration_s": 120, "dt_s": 1,
    "category_shares": {"Normal": 0.40, "Ambiguous": 0.30, "Simulated_Disorientation": 0.30},
    "density_levels_pilgrims_per_m2": {"low": 0.5, "medium": 1.5, "high": 3.0},
    "runs_per_density": 20,
    "background_area_side_m": 30.0, "density_cell_m": 3.0,
    "site_side_m": 300.0, "goal_distance_range_m": [40.0, 180.0],
    "free_speed_mean_mps": 1.25, "free_speed_sd_mps": 0.18, "max_speed_mps": 2.0,
    "max_accel_mps2": 0.45, "max_turn_deg_per_s": 75.0,
    "speed_density_gamma": 1.913, "speed_density_rho_max": 5.4,
    "heading_noise_base_deg": 4.0, "heading_noise_per_density_deg": 2.0,
    "crowd_pause_prob_per_density_per_s": 0.008,
    "onset_time_range_s": [40, 110],
    "preonset_duration_range_s": [10, 30],
    "share_weak_precursor": 0.25,
    "precursor_scale": 1.0,
    "lookalike_prob": {"Normal": 0.15, "Ambiguous": 0.30},
    "disoriented_base_type_ambiguous_prob": 3 / 7,
    "observation_windows_s": [10, 20, 30], "prediction_horizon_s": 30,
    "task_a_window_s": 30, "task_a_stride_s": 10, "task_a_min_event_overlap_s": 10,
    "task_b_prediction_times": list(range(29, 90, 5)),
    "stop_speed_threshold_mps": 0.10, "direction_change_threshold_deg": 30.0,
    "split_runs_per_density": {"train": 14, "validation": 3, "test": 3},
}

# ======================================================================
# Cell 3: The simulator
# ======================================================================
def speed_factor(rho, cfg):
    rho = np.clip(rho, 0.05, cfg["speed_density_rho_max"] - 0.01)
    f = 1 - np.exp(-cfg["speed_density_gamma"] * (1 / rho - 1 / cfg["speed_density_rho_max"]))
    return np.clip(f, 0.08, 1.0)

def simulate_background(rho_nominal, rng, cfg):
    """Moves unrecorded background pilgrims in a repeating square. Returns local density maps, one per second."""
    side, cell = cfg["background_area_side_m"], cfg["density_cell_m"]
    n_cells = int(side / cell)
    n = int(round(rho_nominal * side * side))
    pos = rng.uniform(0, side, size=(n, 2))
    heading = rng.uniform(-np.pi, np.pi, n)
    v0 = np.clip(rng.normal(cfg["free_speed_mean_mps"], cfg["free_speed_sd_mps"], n), 0.5, cfg["max_speed_mps"])
    maps = np.zeros((cfg["duration_s"], n_cells, n_cells))
    for t in range(cfg["duration_s"]):
        ij = np.minimum((pos / cell).astype(int), n_cells - 1)
        flat = ij[:, 0] * n_cells + ij[:, 1]
        counts = np.bincount(flat, minlength=n_cells * n_cells).reshape(n_cells, n_cells)
        dens = counts / (cell * cell)
        maps[t] = dens
        v = v0 * speed_factor(rho_nominal, cfg)   # background speed follows the run's density level
        heading += rng.normal(0, 0.2, n)
        pos = (pos + np.c_[v * np.cos(heading), v * np.sin(heading)]) % side
    return maps

def new_goal(x, y, rng, cfg):
    low, high = cfg["goal_distance_range_m"]
    site = cfg["site_side_m"]
    for _ in range(100):
        d, a = rng.uniform(low, high), rng.uniform(-np.pi, np.pi)
        gx, gy = x + d * np.cos(a), y + d * np.sin(a)
        if 0 <= gx <= site and 0 <= gy <= site:
            return gx, gy
    return site - x, site - y

AMBIGUOUS_EVENTS = {"Valid_Stop": (8, 25), "Valid_Opposite_Walk": (8, 20), "Detour": (10, 30),
                    "Goal_Change": (4, 8), "Crowd_Wait": (10, 25), "Brief_Turn_Back": (4, 10)}
DISORIENTED_MODES = ["Disoriented_Backtrack", "Disoriented_Wander", "Disoriented_Hesitate",
                     "Disoriented_Away", "Disoriented_Partial_Resume"]

def place_event(states, name, duration, rng, low, high):
    """Puts one event in a free part of the [low, high) time range. Returns True if it fitted."""
    if high - low <= duration:
        return False
    for _ in range(20):
        start = int(rng.integers(low, high - duration))
        if all(s == "Walking" for s in states[start:start + duration]):
            states[start:start + duration] = [name] * duration
            return True
    return False

def build_schedule(category, rng, cfg):
    """Decides, before walking starts, what each pilgrim does in each second.
    Before onset, every pilgrim follows the SAME random process. A disoriented pilgrim is an ordinary
    pilgrim (of Normal or Ambiguous type) who later changes, so ordinary behaviour cannot reveal the label."""
    T = cfg["duration_s"]
    states = ["Walking"] * T
    if category == "Simulated_Disorientation":
        base_type = "Ambiguous" if rng.random() < cfg["disoriented_base_type_ambiguous_prob"] else "Normal"
    else:
        base_type = category
    # ordinary slow-downs and pauses
    for _ in range(rng.poisson(1.0)):
        place_event(states, "Slowing", int(rng.integers(3, 9)), rng, 0, T)
    for _ in range(rng.poisson(0.5)):
        place_event(states, "Brief_Stop", int(rng.integers(2, 6)), rng, 0, T)
    # ambiguous events
    if base_type == "Ambiguous":
        names = list(AMBIGUOUS_EVENTS)
        for _ in range(int(rng.integers(1, 4))):
            name = names[rng.integers(len(names))]
            lo, hi = AMBIGUOUS_EVENTS[name]
            place_event(states, name, int(rng.integers(lo, hi + 1)), rng, 0, T)
    # look-alike hesitation: same early signs, but disorientation never follows
    look_start, look_end, look = T + 1, T + 1, 0.0
    if rng.random() < cfg["lookalike_prob"][base_type]:
        look = rng.uniform(0.3, 1.0)
        n = int(rng.integers(10, 31))
        look_start = int(rng.integers(0, T - n))
        look_end = look_start + n
    onset, pre_start, strength = -1, T + 1, 0.0
    if category == "Simulated_Disorientation":
        onset = int(rng.integers(cfg["onset_time_range_s"][0], cfg["onset_time_range_s"][1] + 1))
        dur = int(rng.integers(cfg["preonset_duration_range_s"][0], cfg["preonset_duration_range_s"][1] + 1))
        pre_start = max(onset - dur, 0)
        weak = rng.random() < cfg["share_weak_precursor"]
        strength = rng.uniform(0.0, 0.2) if weak else rng.uniform(0.3, 1.0)
        look_end = min(look_end, onset)
        weights = rng.dirichlet(np.ones(len(DISORIENTED_MODES)))   # every pilgrim has a personal mix
        t = onset
        while t < T:
            mode = DISORIENTED_MODES[rng.choice(len(DISORIENTED_MODES), p=weights)]
            d = int(rng.integers(3, 16))
            states[t:min(t + d, T)] = [mode] * (min(t + d, T) - t)
            t += d
    # state names for seconds with no other event
    for t in range(T):
        if states[t] == "Walking":
            if pre_start <= t < onset:
                states[t] = "Pre_Onset"
            elif look_start <= t < look_end:
                states[t] = "Lookalike_Hesitation"
    return states, onset, pre_start, strength, look_start, look_end, look

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def simulate_pilgrim(pid, run_id, level, category, dens_maps, rng, cfg):
    T = cfg["duration_s"]
    scale = cfg["precursor_scale"]
    side, cell = cfg["background_area_side_m"], cfg["density_cell_m"]
    n_cells = dens_maps.shape[1]
    states, onset, pre_start, strength, look_start, look_end, look = build_schedule(category, rng, cfg)
    PLAIN_WALK = ("Walking", "Pre_Onset", "Lookalike_Hesitation")
    site = cfg["site_side_m"]
    x, y = rng.uniform(20, site - 20, 2)
    gx, gy = new_goal(x, y, rng, cfg)
    v_free = float(np.clip(rng.normal(cfg["free_speed_mean_mps"], cfg["free_speed_sd_mps"]), 0.6, cfg["max_speed_mps"]))
    theta = np.arctan2(gy - y, gx - x)
    rho = dens_maps[0][min(int((x % side) / cell), n_cells - 1), min(int((y % side) / cell), n_cells - 1)]
    v = v_free * float(speed_factor(max(rho, 0.05), cfg))
    wander, pause_left, arrive_left = 0.0, 0, 0
    event_offset = rng.choice([-1, 1]) * rng.uniform(np.radians(50), np.radians(85))
    mode_heading = theta
    max_turn, max_acc = np.radians(cfg["max_turn_deg_per_s"]), cfg["max_accel_mps2"]
    rows = []
    speed_now, dir_now = v, theta
    for t in range(T):
        i = min(int((x % side) / cell), n_cells - 1)
        j = min(int((y % side) / cell), n_cells - 1)
        rho = 0.7 * rho + 0.3 * dens_maps[t][i, j]
        state = states[t]
        to_goal = np.arctan2(gy - y, gx - x)
        dist = np.hypot(gx - x, gy - y)
        if dist < 2.0:                       # reached destination
            if state in PLAIN_WALK and arrive_left == 0:
                arrive_left = int(rng.integers(5, 16))
            gx_new, gy_new = new_goal(x, y, rng, cfg)
            gx, gy = gx_new, gy_new
            to_goal = np.arctan2(gy - y, gx - x)
        if state in PLAIN_WALK and arrive_left > 0:
            state = "Arrived_Pause"; arrive_left -= 1
        if state in PLAIN_WALK and pause_left == 0 and rng.random() < cfg["crowd_pause_prob_per_density_per_s"] * rho:
            pause_left = int(rng.integers(2, 6))
        if state in PLAIN_WALK and pause_left > 0:
            state = "Crowd_Pause"
        pause_left = max(pause_left - 1, 0)
        rows.append((pid, t, x, y, gx, gy, speed_now, np.degrees(dir_now) % 360, rho, category, state, onset, run_id, level))

        sf, desired, extra_noise = 1.0, to_goal, 0.0
        prev = states[t - 1] if t > 0 else "Walking"
        if state in ("Slowing",): sf = 0.55
        elif state in ("Brief_Stop", "Valid_Stop", "Arrived_Pause", "Crowd_Pause"): sf = 0.0
        elif state == "Crowd_Wait": sf = 0.12
        elif state == "Valid_Opposite_Walk": desired = to_goal + np.pi; sf = 0.85
        elif state == "Brief_Turn_Back": desired = to_goal + np.pi; sf = 0.8
        elif state == "Detour": desired = to_goal + event_offset; sf = 0.9
        elif state == "Goal_Change":
            if prev != "Goal_Change":
                gx, gy = new_goal(x, y, rng, cfg)
                desired = np.arctan2(gy - y, gx - x)
            sf = 0.7
        elif state.startswith("Disoriented"):
            if state != prev:
                mode_heading = theta + rng.normal(0, np.radians(70))
            if state == "Disoriented_Backtrack":
                if state != prev: mode_heading = theta + np.pi
                desired = mode_heading; sf = 0.8
                if rng.random() < 0.15: mode_heading += np.pi
            elif state == "Disoriented_Wander":
                mode_heading += rng.normal(0, np.radians(45)); desired = mode_heading; sf = rng.uniform(0.5, 1.0)
            elif state == "Disoriented_Hesitate":
                sf = 0.0 if rng.random() < 0.55 else 0.35; desired = mode_heading
            elif state == "Disoriented_Away":
                desired = to_goal + np.pi + rng.normal(0, np.radians(30)); sf = 0.85
            else:
                sf = 0.75; extra_noise = np.radians(15)

        # early signs are added on top of whatever the pilgrim is doing
        early = 0.0
        if pre_start <= t < onset:
            early = max(early, strength * scale * (t - pre_start + 1) / max(onset - pre_start, 1))
        if look_start <= t < look_end:
            early = max(early, look * scale * (t - look_start + 1) / max(look_end - look_start, 1))
        if early > 0:
            sf *= 1 - 0.45 * early
            extra_noise += np.radians(25) * early
            if rng.random() < 0.15 * early:
                sf = min(sf, 0.1)
        sig = np.radians(cfg["heading_noise_base_deg"] + cfg["heading_noise_per_density_deg"] * rho) + extra_noise
        wander = 0.8 * wander + rng.normal(0, sig)
        theta = wrap(theta + np.clip(wrap(desired + wander - theta), -max_turn, max_turn))
        target_v = v_free * float(speed_factor(max(rho, 0.05), cfg)) * sf * (1 + rng.normal(0, 0.05))
        target_v = float(np.clip(target_v, 0, cfg["max_speed_mps"]))
        v = v + float(np.clip(target_v - v, -max_acc, max_acc))
        v = max(v, 0.0)
        dx, dy = v * np.cos(theta), v * np.sin(theta)
        x, y = x + dx, y + dy
        speed_now, dir_now = v, theta
    return rows

COLUMNS = ["Pilgrim_ID", "Time", "X", "Y", "Goal_X", "Goal_Y", "Speed", "Direction", "Crowd_Density",
           "Movement_Category", "Simulation_State", "Disorientation_Onset", "Run_ID", "Density_Level"]

def generate_dataset(cfg, seed):
    rng = np.random.default_rng(seed)
    levels = cfg["density_levels_pilgrims_per_m2"]
    runs = [(level, r) for level in levels for r in range(cfg["runs_per_density"])]
    n = cfg["n_pilgrims"]
    cats = []
    for c, share in cfg["category_shares"].items():
        cats += [c] * int(round(share * n))
    # deal categories evenly across runs, so category and density stay independent
    per_cat = {c: [] for c in cfg["category_shares"]}
    for c in cats: per_cat[c].append(c)
    run_cats = [[] for _ in runs]
    order = rng.permutation(len(runs))
    k = 0
    for c in cfg["category_shares"]:
        for item in per_cat[c]:
            run_cats[order[k % len(runs)]].append(item); k += 1
    rows, pid = [], 1
    for idx, (level, r) in enumerate(runs):
        maps = simulate_background(levels[level], rng, cfg)
        run_id = f"{level}_{r:02d}"
        for c in rng.permutation(run_cats[idx]):
            rows.extend(simulate_pilgrim(f"P{pid:04d}", run_id, level, str(c), maps, rng, cfg))
            pid += 1
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ["X", "Y", "Goal_X", "Goal_Y", "Speed", "Direction", "Crowd_Density"]:
        df[col] = df[col].astype(float).round(3)
    return df

# ======================================================================
# Cell 4: Generate the dataset and save it
# ======================================================================
df = generate_dataset(CONFIG, CONFIG["master_seed"])
df.to_csv(f"{OUT}/synthetic_pilgrim_trajectories.csv", index=False)
print(df.shape)
print(df.head())
print(df.groupby("Density_Level")[["Crowd_Density", "Speed"]].mean().round(3))

# ======================================================================
# Cell 5: Validate the dataset
# ======================================================================
def check_trajectories(df, cfg):
    g = df.sort_values(["Pilgrim_ID", "Time"]).groupby("Pilgrim_ID")
    pil = g.first()
    step = np.hypot(g.X.diff(), g.Y.diff())
    accel = g.Speed.diff().abs()
    target = {k: int(round(v * cfg["n_pilgrims"])) for k, v in cfg["category_shares"].items()}
    expected_time = list(range(cfg["duration_s"]))
    state_ok = True
    for pid, p in df.groupby("Pilgrim_ID"):
        p = p.sort_values("Time")
        s, o = p.Simulation_State.astype(str), int(p.Disorientation_Onset.iloc[0])
        dis = s.str.startswith("Disoriented")
        pre = s == "Pre_Onset"
        if p.Movement_Category.iloc[0] == "Simulated_Disorientation":
            state_ok &= o >= 0 and dis.equals(p.Time >= o) and (p.Time[pre] < o).all()
        else:
            state_ok &= o == -1 and not dis.any() and not pre.any()
    rows = [("No missing values", int(df.isna().sum().sum()) == 0, int(df.isna().sum().sum())),
            ("Row count", len(df) == cfg["n_pilgrims"] * cfg["duration_s"], len(df)),
            ("Unique Pilgrim_ID", df.Pilgrim_ID.nunique() == cfg["n_pilgrims"], df.Pilgrim_ID.nunique()),
            ("No duplicate Pilgrim_ID and Time", not df.duplicated(["Pilgrim_ID", "Time"]).any(), int(df.duplicated(["Pilgrim_ID", "Time"]).sum())),
            ("Continuous timestamps", all(list(t) == expected_time for t in g.Time.apply(list)), "0 to 119 for all"),
            ("Realistic speeds", df.Speed.between(0, cfg["max_speed_mps"]).all(), f"max {df.Speed.max():.2f} m/s"),
            ("No position jumps", step.max() <= cfg["max_speed_mps"] + 0.01, f"max step {step.max():.2f} m"),
            ("Smooth speed changes", accel.max() <= cfg["max_accel_mps2"] + 0.01, f"max change {accel.max():.2f} m/s"),
            ("Valid state transitions", bool(state_ok), "onset, pre-onset and category agree"),
            ("Correct class distribution", pil.Movement_Category.value_counts().to_dict() == target, pil.Movement_Category.value_counts().to_dict()),
            ("Consistent labels per pilgrim", (g.Movement_Category.nunique() == 1).all() and (g.Run_ID.nunique() == 1).all(), "one category and run per pilgrim")]
    return rows
val_rows = check_trajectories(df, CONFIG)
for name, ok, detail in val_rows:
    print("PASS" if ok else "FAIL", "|", name, "|", detail)
pil = df.groupby("Pilgrim_ID").first()
print(pd.crosstab(pil.Density_Level, pil.Movement_Category))

# ======================================================================
# Cell 6: Plot example trajectories
# ======================================================================
rng_plot = np.random.default_rng(CONFIG["master_seed"])
cats = list(CONFIG["category_shares"])
levels = list(CONFIG["density_levels_pilgrims_per_m2"])
fig, axes = plt.subplots(3, 3, figsize=(13, 12))
for r, cat in enumerate(cats):
    for c, lev in enumerate(levels):
        ax = axes[r, c]
        ids = pil[(pil.Movement_Category == cat) & (pil.Density_Level == lev)].index.to_numpy()
        for pid in rng_plot.choice(ids, 4, replace=False):
            p = df[df.Pilgrim_ID == pid]
            line, = ax.plot(p.X, p.Y, lw=1.2)
            ax.scatter(p.X.iloc[0], p.Y.iloc[0], color=line.get_color(), marker="o", s=25)
            o = p.Disorientation_Onset.iloc[0]
            if o >= 0:
                q = p[p.Time == o]
                ax.scatter(q.X, q.Y, color="black", marker="x", s=45, zorder=5)
        ax.set_title(f"{cat}, {lev} density", fontsize=10)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.axis("equal")
fig.suptitle("Synthetic example trajectories (circle = start, x = simulated onset)")
fig.tight_layout()
fig.savefig(f"{OUT}/example_trajectories.png", dpi=170)
plt.show()

# ======================================================================
# Cell 7: Feature extraction
# ======================================================================
FOUR_FEATURES = ["speed_variation", "direction_changes", "dwell_time", "path_tortuosity"]
EXTENDED_FEATURES = ["average_speed", "speed_variation", "total_path_length", "net_displacement", "straightness_ratio",
                     "average_turning_angle", "direction_changes", "stop_frequency", "dwell_time",
                     "progress_toward_destination", "average_crowd_density"]
ALL_FEATURES = sorted(set(FOUR_FEATURES) | set(EXTENDED_FEATURES))
FORBIDDEN_INPUTS = {"Movement_Category", "Simulation_State", "Disorientation_Onset", "Pilgrim_ID", "Run_ID", "Density_Level",
                    "Time", "End_Time", "Goal_X", "Goal_Y", "Target", "Split", "Task", "Window_Seconds", "Lead_Time"}

def window_features(x, y, gx, gy, density, cfg):
    """All inputs cover the observation window only; nothing after its last second is passed in."""
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    path, net = step.sum(), np.hypot(x[-1] - x[0], y[-1] - y[0])
    stopped = step < cfg["stop_speed_threshold_mps"]
    heading = np.arctan2(dy[~stopped], dx[~stopped])
    turns = np.degrees(np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi)) if heading.size > 1 else np.zeros(1)
    straight = net / path if path > 0.1 else 0.0
    d_now = np.hypot(gx[:-1] - x[:-1], gy[:-1] - y[:-1])
    d_next = np.hypot(gx[:-1] - x[1:], gy[:-1] - y[1:])
    return {"average_speed": step.mean(), "speed_variation": step.std(), "total_path_length": path, "net_displacement": net,
            "straightness_ratio": straight, "path_tortuosity": 1 - straight, "average_turning_angle": turns.mean(),
            "direction_changes": int((turns > cfg["direction_change_threshold_deg"]).sum()),
            "stop_frequency": int((stopped[1:] & ~stopped[:-1]).sum()), "dwell_time": int(stopped.sum()),
            "progress_toward_destination": (d_now - d_next).sum() / path if path > 0.1 else 0.0,
            "average_crowd_density": density.mean()}

def pilgrim_arrays(df):
    out = {}
    for pid, g in df.sort_values(["Pilgrim_ID", "Time"]).groupby("Pilgrim_ID", sort=False):
        out[pid] = dict(x=g.X.to_numpy(), y=g.Y.to_numpy(), gx=g.Goal_X.to_numpy(), gy=g.Goal_Y.to_numpy(),
                        rho=g.Crowd_Density.to_numpy(), state=g.Simulation_State.to_numpy(),
                        cat=g.Movement_Category.iloc[0], onset=int(g.Disorientation_Onset.iloc[0]),
                        run=g.Run_ID.iloc[0], level=g.Density_Level.iloc[0])
    return out

AMBIGUOUS_STATES = set(AMBIGUOUS_EVENTS)

def make_samples_task_a(arr, cfg):
    W, rows = cfg["task_a_window_s"], []
    for pid, p in arr.items():
        for end in range(W - 1, cfg["duration_s"], cfg["task_a_stride_s"]):
            start = end - W + 1
            if p["cat"] == "Normal":
                label = "Normal"
            elif p["cat"] == "Ambiguous":
                overlap = sum(s in AMBIGUOUS_STATES for s in p["state"][start:end + 1])
                if overlap < cfg["task_a_min_event_overlap_s"]:
                    continue
                label = "Ambiguous"
            else:
                if start < p["onset"]:
                    continue
                label = "Simulated_Disorientation"
            f = window_features(p["x"][start:end+1], p["y"][start:end+1], p["gx"][start:end+1], p["gy"][start:end+1], p["rho"][start:end+1], cfg)
            rows.append({"Task": "A", "Window_Seconds": W, "Pilgrim_ID": pid, "Run_ID": p["run"], "Density_Level": p["level"],
                         "End_Time": end, "Target": label, "Lead_Time": -1, **f})
    return pd.DataFrame(rows)

def make_samples_task_b(arr, cfg, windows=None):
    H, rows = cfg["prediction_horizon_s"], []
    windows = windows or cfg["observation_windows_s"]
    for pid, p in arr.items():
        for end in cfg["task_b_prediction_times"]:
            if p["onset"] >= 0 and end >= p["onset"]:
                continue                      # already disoriented: excluded, as in Phase 1
            positive = p["onset"] >= 0 and end < p["onset"] <= end + H
            for W in windows:
                start = end - W + 1
                f = window_features(p["x"][start:end+1], p["y"][start:end+1], p["gx"][start:end+1], p["gy"][start:end+1], p["rho"][start:end+1], cfg)
                rows.append({"Task": "B", "Window_Seconds": W, "Pilgrim_ID": pid, "Run_ID": p["run"], "Density_Level": p["level"],
                             "End_Time": end, "Target": int(positive), "Lead_Time": (p["onset"] - end) if positive else -1, **f})
    return pd.DataFrame(rows)

def split_runs(df, cfg, seed):
    rng = np.random.default_rng(seed)
    split = {}
    n = cfg["split_runs_per_density"]
    for level, g in df.groupby("Density_Level"):
        runs = rng.permutation(sorted(g.Run_ID.unique()))
        for i, r in enumerate(runs):
            split[str(r)] = "train" if i < n["train"] else ("validation" if i < n["train"] + n["validation"] else "test")
    return split


arr = pilgrim_arrays(df)
split_map = split_runs(df, CONFIG, CONFIG["master_seed"])
task_a = make_samples_task_a(arr, CONFIG)
task_b = make_samples_task_b(arr, CONFIG)
for t in (task_a, task_b):
    t["Split"] = t.Run_ID.map(split_map)
features_out = pd.concat([task_a.astype({"Target": str}), task_b.astype({"Target": str})], ignore_index=True)
features_out.round(4).to_csv(f"{OUT}/pilgrim_features.csv", index=False)
print(task_a.groupby(["Split", "Target"]).size())
print(task_b[task_b.Window_Seconds == 30].groupby("Split").Target.agg(["mean", "size"]).round(3))

# ======================================================================
# Cell 8: Split check and leakage checks
# ======================================================================
sets = {k: set(task_b[task_b.Split == k].Pilgrim_ID) for k in ("train", "validation", "test")}
pilgrim_split = pil.Run_ID.map(split_map)
window_end_ok = all(arr[r.Pilgrim_ID]["onset"] < 0 or r.End_Time < arr[r.Pilgrim_ID]["onset"]
                    for r in task_b[["Pilgrim_ID", "End_Time"]].itertuples())
pid0 = next(k for k, v in arr.items() if v["onset"] >= 0)
p0 = {k: arr[pid0][k].copy() for k in ("x", "y", "gx", "gy", "rho")}
a0 = window_features(*(p0[k][10:40] for k in ("x", "y", "gx", "gy", "rho")), CONFIG)
p1 = {k: v.copy() for k, v in p0.items()}
p1["x"][40:] += 500; p1["y"][40:] -= 500
a1 = window_features(*(p1[k][10:40] for k in ("x", "y", "gx", "gy", "rho")), CONFIG)
leak_rows = [("No pilgrim in two splits", not (pilgrim_split.groupby(pilgrim_split.index).nunique() > 1).any()
              and not (sets["train"] & sets["test"] or sets["train"] & sets["validation"] or sets["validation"] & sets["test"]), "train/validation/test are disjoint"),
             ("Split sizes", True, pilgrim_split.value_counts().to_dict()),
             ("No forbidden column used as input", not (set(ALL_FEATURES) & FORBIDDEN_INPUTS), "features exclude labels, IDs, onset, time, goals"),
             ("Task B windows end before onset", window_end_ok, "checked on every sample"),
             ("Changing the future does not change features", a0 == a1, "future coordinates altered in a test"),
             ("Same prediction times in both Task B classes", set(task_b[task_b.Target == 1].End_Time) == set(task_b[task_b.Target == 0].End_Time), "time is not an input")]
for name, ok, detail in leak_rows:
    print("PASS" if ok else "FAIL", "|", name, "|", detail)
pd.DataFrame(val_rows + leak_rows, columns=["Check", "Passed", "Detail"]).to_csv(f"{OUT}/validation_report.csv", index=False)

# ======================================================================
# Cell 9: Task A, movement pattern classification (supporting experiment)
# ======================================================================
def rf(leaf=5, seed=350):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf, class_weight="balanced", n_jobs=-1, random_state=seed)
def lr(C=1.0, seed=350):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))
GRID = {"RandomForest": [("min_samples_leaf=%d" % k, (lambda k=k: rf(leaf=k))) for k in (1, 5, 20)],
        "LogisticRegression": [("C=%g" % c, (lambda c=c: lr(C=c))) for c in (0.1, 1.0, 10.0)]}

def best_threshold(y, p):
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])

def fit_binary(train, val, feats, model_name):
    """Choose setting and decision threshold using validation only."""
    best = None
    for tag, make in GRID[model_name]:
        m = make().fit(train[feats], train.Target)
        p = m.predict_proba(val[feats])[:, 1]
        t = best_threshold(val.Target, p)
        score = f1_score(val.Target, p >= t, zero_division=0)
        if best is None or score > best["val_f1"]:
            best = dict(model=m, setting=tag, threshold=t, val_f1=score, val_pr_auc=average_precision_score(val.Target, p))
    return best

def binary_scores(y, p, t):
    pred = p >= t
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
                PR_AUC=average_precision_score(y, p) if len(set(y)) > 1 else np.nan,
                ROC_AUC=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                Positive_Rate=float(np.mean(y)), N=len(y))

def cluster_bootstrap_f1(frame, p, t, seed, n_boot=500):
    """Resamples whole pilgrims, because samples from one pilgrim are not independent."""
    rng = np.random.default_rng(seed)
    y, pred = frame.Target.to_numpy(), (p >= t)
    groups = {k: v for k, v in frame.groupby("Pilgrim_ID").indices.items()}
    ids = list(groups)
    tp = np.array([(y[groups[i]] & pred[groups[i]]).sum() for i in ids])
    fp = np.array([((1 - y[groups[i]]) & pred[groups[i]]).sum() for i in ids])
    fn = np.array([(y[groups[i]] & ~pred[groups[i]]).sum() for i in ids])
    out = np.zeros(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        TP, FP, FN = tp[pick].sum(), fp[pick].sum(), fn[pick].sum()
        out[b] = 2 * TP / max(2 * TP + FP + FN, 1)
    return out

EVAL = []
def add_eval(**kw):
    EVAL.append(kw)

A_tr, A_va, A_te = (task_a[task_a.Split == k] for k in ("train", "validation", "test"))
A_LABELS = ["Normal", "Ambiguous", "Simulated_Disorientation"]
def multi_scores(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=A_LABELS, average="macro", zero_division=0)
    return dict(Accuracy=accuracy_score(y, pred), Precision=p, Recall=r, F1=f, N=len(y))
best_a = {}
for name in GRID:
    best = None
    for tag, make in GRID[name]:
        m = make().fit(A_tr[EXTENDED_FEATURES], A_tr.Target)
        score = f1_score(A_va.Target, m.predict(A_va[EXTENDED_FEATURES]), labels=A_LABELS, average="macro")
        if best is None or score > best[0]:
            best = (score, tag, m)
    best_a[name] = best
    pred = best[2].predict(A_te[EXTENDED_FEATURES])
    add_eval(Experiment="Task A", Model=name, Feature_Set="extended", Window_Seconds=30, Density="all", Class="macro", Setting=best[1], **multi_scores(A_te.Target, pred))
    pp, rr, ff, nn = precision_recall_fscore_support(A_te.Target, pred, labels=A_LABELS, zero_division=0)
    for i, lab in enumerate(A_LABELS):
        add_eval(Experiment="Task A", Model=name, Feature_Set="extended", Window_Seconds=30, Density="all", Class=lab, Setting=best[1], Precision=pp[i], Recall=rr[i], F1=ff[i], N=int(nn[i]))
    for lev in levels:
        s = A_te.Density_Level == lev
        add_eval(Experiment="Task A", Model=name, Feature_Set="extended", Window_Seconds=30, Density=lev, Class="macro", Setting=best[1], **multi_scores(A_te[s].Target, pred[s.to_numpy()]))
for strategy in ("most_frequent", "stratified"):
    d = DummyClassifier(strategy=strategy, random_state=CONFIG["master_seed"]).fit(A_tr[EXTENDED_FEATURES], A_tr.Target)
    add_eval(Experiment="Task A", Model=f"Dummy_{strategy}", Feature_Set="none", Window_Seconds=30, Density="all", Class="macro", Setting="", **multi_scores(A_te.Target, d.predict(A_te[EXTENDED_FEATURES])))
print(pd.DataFrame(EVAL).query("Density == 'all'")[["Model", "Class", "Accuracy", "Precision", "Recall", "F1", "N"]].round(3).to_string(index=False))

# ======================================================================
# Cell 10: Is there a defensible signal before onset? (gate for Task B)
# ======================================================================
def prepare_b(scale, seed):
    cfg = dict(CONFIG, precursor_scale=scale)
    d = generate_dataset(cfg, seed)
    a = pilgrim_arrays(d)
    s = split_runs(d, cfg, seed)
    b = make_samples_task_b(a, cfg)
    b["Split"] = b.Run_ID.map(s)
    return b

# size of the difference between samples close to onset and comparable negatives (Cohen's d)
w10 = task_b[task_b.Window_Seconds == 10]
near = w10[(w10.Target == 1) & (w10.Lead_Time <= 10)]
negs = w10[w10.Target == 0]
signal = []
for f in ALL_FEATURES:
    sd = np.sqrt((near[f].var() + negs[f].var()) / 2)
    row = {"Feature": f, "All_densities": (near[f].mean() - negs[f].mean()) / sd}
    for lev in levels:
        n1, n0 = near[near.Density_Level == lev][f], negs[negs.Density_Level == lev][f]
        row[lev] = (n1.mean() - n0.mean()) / np.sqrt((n1.var() + n0.var()) / 2)
    signal.append(row)
signal = pd.DataFrame(signal).round(2)
signal.to_csv(f"{OUT}/early_signal_check.csv", index=False)
print(signal.to_string(index=False))

control = prepare_b(0.0, CONFIG["master_seed"])
gate_rows = []
for fs_name, feats in (("four", FOUR_FEATURES), ("extended", EXTENDED_FEATURES)):
    for W in CONFIG["observation_windows_s"]:
        for model in GRID:
            bc = control[control.Window_Seconds == W]
            ctr, cva = bc[bc.Split == "train"], bc[bc.Split == "validation"]
            mk = rf(leaf=20) if model == "RandomForest" else lr()
            pc = mk.fit(ctr[feats], ctr.Target).predict_proba(cva[feats])[:, 1]
            br = task_b[task_b.Window_Seconds == W]
            rtr, rva = br[br.Split == "train"], br[br.Split == "validation"]
            mk = rf(leaf=20) if model == "RandomForest" else lr()
            pr = mk.fit(rtr[feats], rtr.Target).predict_proba(rva[feats])[:, 1]
            gate_rows.append(dict(Feature_Set=fs_name, Window_Seconds=W, Model=model,
                                  Control_Positive_Rate=cva.Target.mean(), Control_PR_AUC=average_precision_score(cva.Target, pc), Control_ROC_AUC=roc_auc_score(cva.Target, pc),
                                  Real_Positive_Rate=rva.Target.mean(), Real_PR_AUC=average_precision_score(rva.Target, pr), Real_ROC_AUC=roc_auc_score(rva.Target, pr)))
gate = pd.DataFrame(gate_rows)
gate["Control_Near_Chance"] = gate.Control_ROC_AUC.between(0.40, 0.60)
gate["Signal_Above_Margin"] = gate.Real_PR_AUC >= gate.Real_Positive_Rate + 0.05
gate["Gate_Passed"] = gate.Control_Near_Chance & gate.Signal_Above_Margin
gate.round(3).to_csv(f"{OUT}/task_b_gate.csv", index=False)
print(gate.round(3).to_string(index=False))
TASK_B_VALID = bool(gate.Gate_Passed.any())
print("\nTask B may continue:", TASK_B_VALID)
if not TASK_B_VALID:
    print("LIMITATION: no defensible early-warning signal under this design. Report this; do not tune until it passes.")

# ======================================================================
# Cell 11: Task B, future disorientation prediction (RQ1, primary experiment)
# ======================================================================
FITS, PRED = {}, {}
if TASK_B_VALID:
    for fs_name, feats in (("four", FOUR_FEATURES), ("extended", EXTENDED_FEATURES)):
        for W in CONFIG["observation_windows_s"]:
            b = task_b[task_b.Window_Seconds == W]
            tr, va, te = (b[b.Split == k] for k in ("train", "validation", "test"))
            for model in GRID:
                fit = fit_binary(tr, va, feats, model)
                p = fit["model"].predict_proba(te[feats])[:, 1]
                FITS[(fs_name, W, model)] = fit
                PRED[(fs_name, W, model)] = (te, p)
                add_eval(Experiment="Task B", Model=model, Feature_Set=fs_name, Window_Seconds=W, Density="all", Class="positive", Setting=fit["setting"], Threshold=fit["threshold"], Validation_F1=fit["val_f1"], Validation_PR_AUC=fit["val_pr_auc"], **binary_scores(te.Target, p, fit["threshold"]))
                for lev in levels:
                    s = (te.Density_Level == lev).to_numpy()
                    add_eval(Experiment="Task B", Model=model, Feature_Set=fs_name, Window_Seconds=W, Density=lev, Class="positive", Setting=fit["setting"], Threshold=fit["threshold"], **binary_scores(te.Target[s], p[s], fit["threshold"]))
            for strategy in ("most_frequent", "stratified", "always_positive"):
                d = (DummyClassifier(strategy="constant", constant=1) if strategy == "always_positive"
                     else DummyClassifier(strategy=strategy, random_state=CONFIG["master_seed"])).fit(tr[feats], tr.Target)
                pd_ = d.predict(te[feats]).astype(float)
                if fs_name == "four":
                    add_eval(Experiment="Task B", Model=f"Dummy_{strategy}", Feature_Set="none", Window_Seconds=W, Density="all", Class="positive", Setting="", Threshold=0.5, **binary_scores(te.Target, pd_, 0.5))
    res_b = pd.DataFrame(EVAL).query("Experiment == 'Task B' and Density == 'all'")
    print(res_b[["Feature_Set", "Window_Seconds", "Model", "Setting", "Threshold", "Accuracy", "Precision", "Recall", "F1", "PR_AUC", "ROC_AUC", "Positive_Rate"]].round(3).to_string(index=False))

    # configuration chosen using validation only, for detailed reports
    key_best = max(FITS, key=lambda k: FITS[k]["val_pr_auc"])
    print("\nBest on validation:", key_best)
    te, p = PRED[key_best]
    th = FITS[key_best]["threshold"]
    lead = te[te.Target == 1].assign(pred=p[te.Target.to_numpy() == 1] >= th)
    lead["Lead_Bin"] = pd.cut(lead.Lead_Time, [0, 10, 20, 30], labels=["1-10 s", "11-20 s", "21-30 s"])
    lead_tab = lead.groupby("Lead_Bin", observed=True).pred.agg(Recall="mean", N="size").round(3)
    print(lead_tab)
    lead_tab.to_csv(f"{OUT}/recall_by_lead_time.csv")

    # label-shuffling check, by pilgrim
    b = task_b[task_b.Window_Seconds == key_best[1]]
    feats = FOUR_FEATURES if key_best[0] == "four" else EXTENDED_FEATURES
    tr, te2 = b[b.Split == "train"].copy(), b[b.Split == "test"]
    tr["Target"] = np.random.default_rng(1).permutation(tr.Target.to_numpy())
    pshuf = rf(leaf=20).fit(tr[feats], tr.Target).predict_proba(te2[feats])[:, 1]
    print("Shuffled-label test ROC-AUC (should be close to 0.5):", round(roc_auc_score(te2.Target, pshuf), 3))

# ======================================================================
# Cell 12: Feature importance (RQ2)
# ======================================================================
imp_rows = []
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
def perm(model, frame, feats, y, score):
    r = permutation_importance(model, frame[feats], y, n_repeats=10, random_state=CONFIG["master_seed"], scoring=score, n_jobs=-1)
    return r.importances_mean, r.importances_std
m_a = best_a["RandomForest"][2]
mean, sd = perm(m_a, A_te, EXTENDED_FEATURES, A_te.Target, "f1_macro")
for f, a, b_ in zip(EXTENDED_FEATURES, mean, sd):
    imp_rows.append(dict(Experiment="Task A", Model="RandomForest", Feature_Set="extended", Window_Seconds=30, Feature=f, Permutation_Importance=a, SD=b_))
o = np.argsort(mean)
axes[0].barh(np.array(EXTENDED_FEATURES)[o], mean[o], xerr=sd[o], color="#176f7a")
axes[0].set_title("Task A: Random Forest, 30 s window"); axes[0].set_xlabel("Drop in macro-F1 when shuffled")
if TASK_B_VALID:
    for fs_name, feats in (("four", FOUR_FEATURES), ("extended", EXTENDED_FEATURES)):
        for W in CONFIG["observation_windows_s"]:
            for model in GRID:
                te, _ = PRED[(fs_name, W, model)]
                est = FITS[(fs_name, W, model)]["model"]
                mean, sd = perm(est, te, feats, te.Target, "average_precision")
                gini = est.feature_importances_ if model == "RandomForest" else [np.nan] * len(feats)
                for f, a, b_, g in zip(feats, mean, sd, gini):
                    imp_rows.append(dict(Experiment="Task B", Model=model, Feature_Set=fs_name, Window_Seconds=W, Feature=f, Permutation_Importance=a, SD=b_, RF_Builtin_Importance=g))
    imp = pd.DataFrame(imp_rows)
    four = imp[(imp.Experiment == "Task B") & (imp.Feature_Set == "four")]
    table = four.pivot_table(index="Feature", columns=["Model", "Window_Seconds"], values="Permutation_Importance")
    print(table.round(4).to_string())
    pv = four.groupby(["Feature", "Model"]).Permutation_Importance.mean().unstack()
    pv.plot.barh(ax=axes[1], color=["#c8833a", "#176f7a"])
    axes[1].set_title("Task B: Phase 1 four features\n(mean across 10, 20, 30 s windows)"); axes[1].set_xlabel("Drop in PR-AUC when shuffled"); axes[1].set_ylabel("")
    k = key_best
    s = imp[(imp.Experiment == "Task B") & (imp.Feature_Set == k[0]) & (imp.Window_Seconds == k[1]) & (imp.Model == k[2])].sort_values("Permutation_Importance")
    axes[2].barh(s.Feature, s.Permutation_Importance, xerr=s.SD, color="#176f7a")
    axes[2].set_title(f"Task B: {k[2]}, {k[0]} features, {k[1]} s\n(best on validation)"); axes[2].set_xlabel("Drop in PR-AUC when shuffled")
else:
    imp = pd.DataFrame(imp_rows)
imp.round(5).to_csv(f"{OUT}/feature_importance_values.csv", index=False)
fig.tight_layout(); fig.savefig(f"{OUT}/feature_importance.png", dpi=170); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
def draw_cm(ax, cm, labels, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right"); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
cm_a = confusion_matrix(A_te.Target, m_a.predict(A_te[EXTENDED_FEATURES]), labels=A_LABELS)
draw_cm(axes[0], cm_a, ["Normal", "Ambiguous", "Sim_Disorient."], "Task A: Random Forest (test windows)")
print("Task A confusion matrix\n", cm_a)
if TASK_B_VALID:
    te, p = PRED[key_best]
    cm_b = confusion_matrix(te.Target, p >= FITS[key_best]["threshold"], labels=[0, 1])
    draw_cm(axes[1], cm_b, ["No onset in 30 s", "Onset in 30 s"], f"Task B: {key_best[2]}, {key_best[0]}, {key_best[1]} s (test)")
    print("Task B confusion matrix\n", cm_b)
else:
    axes[1].axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=170); plt.show()

# ======================================================================
# Cell 13: Crowd density (RQ3 and the Phase 1 hypotheses)
# ======================================================================
rq3 = []
if TASK_B_VALID:
    for (fs_name, W, model), (te, p) in PRED.items():
        t = FITS[(fs_name, W, model)]["threshold"]
        boots, row = {}, dict(Feature_Set=fs_name, Window_Seconds=W, Model=model)
        for lev in levels:
            s = (te.Density_Level == lev).to_numpy()
            fr, ps = te[s], p[s]
            row[f"F1_{lev}"] = f1_score(fr.Target, ps >= t, zero_division=0)
            row[f"AlwaysPositive_F1_{lev}"] = f1_score(fr.Target, np.ones(len(fr), dtype=int), zero_division=0)
            row[f"PR_AUC_{lev}"] = average_precision_score(fr.Target, ps)
            boots[lev] = cluster_bootstrap_f1(fr, ps, t, CONFIG["master_seed"])
        diff = boots["low"] - boots["high"]
        row["Low_minus_High"] = row["F1_low"] - row["F1_high"]
        row["CI_Low"], row["CI_High"] = np.percentile(diff, [2.5, 97.5])
        row["H1_Supported"] = bool(row["F1_low"] > row["F1_medium"] > row["F1_high"] and row["CI_Low"] > 0)
        rq3.append(row)
    rq3 = pd.DataFrame(rq3)
    rq3.round(3).to_csv(f"{OUT}/rq3_density_results.csv", index=False)
    print(rq3[["Feature_Set", "Window_Seconds", "Model", "F1_low", "F1_medium", "F1_high", "Low_minus_High", "CI_Low", "CI_High", "H1_Supported"]].round(3).to_string(index=False))
    print("Always-positive F1 by density:", rq3.iloc[0][["AlwaysPositive_F1_low", "AlwaysPositive_F1_medium", "AlwaysPositive_F1_high"]].round(3).to_dict())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, fs_name in zip(axes, ("four", "extended")):
        for (W, model), g in rq3[rq3.Feature_Set == fs_name].groupby(["Window_Seconds", "Model"]):
            ax.plot(levels, g[[f"F1_{l}" for l in levels]].to_numpy().ravel(), marker="o", ls="-" if model == "RandomForest" else "--", label=f"{model[:2].upper()} {W}s")
        ax.plot(levels, rq3.iloc[0][[f"AlwaysPositive_F1_{l}" for l in levels]].to_numpy(dtype=float), color="black", lw=2, ls=":", label="Always positive")
        ax.set_title(f"{fs_name} features"); ax.set_xlabel("Simulated crowd density"); ax.set_ylabel("Test F1"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/f1_by_density.png", dpi=170); plt.show()

# ======================================================================
# Cell 14: Robustness checks
# ======================================================================
robust, dose = [], []
if TASK_B_VALID:
    t0 = time.time()
    for seed in [101, 202, 303, 404, 505]:
        b_all = prepare_b(1.0, seed)
        for fs_name, feats in (("four", FOUR_FEATURES), ("extended", EXTENDED_FEATURES)):
            for W in CONFIG["observation_windows_s"]:
                b = b_all[b_all.Window_Seconds == W]
                tr, va, te = (b[b.Split == k] for k in ("train", "validation", "test"))
                for model, make in (("RandomForest", lambda: rf(leaf=20)), ("LogisticRegression", lr)):
                    m = make().fit(tr[feats], tr.Target)
                    th = best_threshold(va.Target, m.predict_proba(va[feats])[:, 1])
                    p = m.predict_proba(te[feats])[:, 1]
                    row = dict(Seed=seed, Feature_Set=fs_name, Window_Seconds=W, Model=model, **binary_scores(te.Target, p, th))
                    row["Always_Positive_F1"] = f1_score(te.Target, np.ones(len(te), dtype=int))
                    for lev in levels:
                        s = (te.Density_Level == lev).to_numpy()
                        row[f"F1_{lev}"] = f1_score(te.Target[s], p[s] >= th, zero_division=0)
                    robust.append(row)
        print("seed", seed, "done", round(time.time() - t0), "s")
    robust = pd.DataFrame(robust)
    robust.round(4).to_csv(f"{OUT}/robustness_by_seed.csv", index=False)
    summary = robust.groupby(["Feature_Set", "Window_Seconds", "Model"])[["F1", "Always_Positive_F1", "PR_AUC", "Positive_Rate", "ROC_AUC", "F1_low", "F1_medium", "F1_high"]].agg(["mean", "std"]).round(3)
    summary.to_csv(f"{OUT}/robustness_summary.csv")
    print(summary[[("F1", "mean"), ("F1", "std"), ("Always_Positive_F1", "mean"), ("PR_AUC", "mean"), ("Positive_Rate", "mean"), ("F1_low", "mean"), ("F1_medium", "mean"), ("F1_high", "mean")]].to_string())
    fs_name, W, model = key_best
    feats = FOUR_FEATURES if fs_name == "four" else EXTENDED_FEATURES
    for scale in [0.0, 0.25, 0.5, 1.0]:
        b_all = prepare_b(scale, CONFIG["master_seed"])
        b = b_all[b_all.Window_Seconds == W]
        tr, te = b[b.Split == "train"], b[b.Split == "test"]
        p = (rf(leaf=20) if model == "RandomForest" else lr()).fit(tr[feats], tr.Target).predict_proba(te[feats])[:, 1]
        dose.append(dict(Precursor_Scale=scale, PR_AUC=average_precision_score(te.Target, p), ROC_AUC=roc_auc_score(te.Target, p), Positive_Rate=te.Target.mean()))
    dose = pd.DataFrame(dose)
    dose.round(3).to_csv(f"{OUT}/signal_strength_check.csv", index=False)
    print(dose.round(3).to_string(index=False))

# ======================================================================
# Cell 15: Save the evaluation table and parameters
# ======================================================================
evaluation = pd.DataFrame(EVAL)
evaluation.round(4).to_csv(f"{OUT}/model_evaluation.csv", index=False)
params = dict(CONFIG)
params.update({
    "dataset_type": "SYNTHETIC. No real pilgrim data.",
    "generator": "Custom Python agent-style simulation, code generated with the assistance of Claude (Anthropic) and reviewed by the team.",
    "library_versions": VERSIONS,
    "robustness_seeds": [101, 202, 303, 404, 505],
    "ambiguous_events_duration_range_s": AMBIGUOUS_EVENTS,
    "disoriented_modes": DISORIENTED_MODES,
    "feature_sets": {"four_phase1": FOUR_FEATURES, "extended": EXTENDED_FEATURES},
    "models": {"RandomForest": "200 trees, balanced class weights, min_samples_leaf chosen on validation from 1, 5, 20",
               "LogisticRegression": "standard scaling fitted on training data, balanced class weights, C chosen on validation from 0.1, 1, 10"},
    "assumptions": ["Pre-onset signs and disorientation rules are modelling assumptions, not proven signs of real disorientation.",
                    "Speed falls with density using a Weidmann-type speed-density curve. The team must verify the source before citing it.",
                    "Density is a stated setting. Background pilgrims affect recorded pilgrims only through local density and do not collide or avoid each other.",
                    "Movement categories were defined manually by the research team.",
                    "The site has no walls, obstacles, ritual routes, groups, or age and mobility differences."],
    "category_definitions": {"Normal": "Goal-directed walking with ordinary turns and pauses.",
                             "Ambiguous": "Unusual but valid behaviour that must not be treated as disorientation.",
                             "Simulated_Disorientation": "Hypothetical behavioural state produced by simulator rules; not verified real-world disorientation."},
    "task_b_gate_passed": TASK_B_VALID})
with open(f"{OUT}/simulation_parameters.json", "w") as fh:
    json.dump(params, fh, indent=2)
required = ["synthetic_pilgrim_trajectories.csv", "pilgrim_features.csv", "simulation_parameters.json", "model_evaluation.csv",
            "confusion_matrix.png", "feature_importance.png", "example_trajectories.png"]
for f in required:
    print("OK     " if os.path.exists(f"{OUT}/{f}") else "MISSING", f)
print("Total run time:", round((time.time() - START) / 60, 1), "minutes")

# ======================================================================
# Cell 16: Export everything to a ZIP
# ======================================================================
zip_name = "pilgrim_outputs.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), arcname=f)
print("Created", zip_name, round(os.path.getsize(zip_name) / 1e6, 1), "MB")
try:
    from google.colab import files
    files.download(zip_name)
except ImportError:
    print("Not running in Colab: find the ZIP in the working folder.")


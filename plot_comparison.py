import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
log_root = "logs"  # The folder shown in your image
experiments = {
    "Baseline (Blind)": "baseline_test",
    "God Mode (Privileged)": "god_mode_test"
}
metric_to_plot = "Train/mean_reward"
# ---------------------

def get_log_file(experiment_name):
    # 1. Enter the experiment folder (e.g., logs/baseline_test)
    exp_path = os.path.join(log_root, experiment_name)
    if not os.path.exists(exp_path):
        print(f"Error: Folder {exp_path} not found.")
        return None

    # 2. Find the timestamped sub-folder (e.g., Feb05_...)
    subdirs = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
    if not subdirs:
        print(f"Error: No run folder found inside {exp_path}")
        return None
    
    # Pick the last one (most recent run)
    subdirs.sort()
    latest_run = subdirs[-1]
    full_path = os.path.join(exp_path, latest_run)
    
    # 3. Find the events file inside
    event_files = [f for f in os.listdir(full_path) if "tfevents" in f]
    if not event_files:
        print(f"Error: No event file found in {full_path}")
        return None
        
    return os.path.join(full_path, event_files[0])

def get_data(experiment_name):
    path = get_log_file(experiment_name)
    if path is None:
        return [], []
    
    print(f"Loading data from: {path}")
    ea = EventAccumulator(path)
    ea.Reload()
    
    if metric_to_plot not in ea.Tags()['scalars']:
        print(f"Warning: Metric '{metric_to_plot}' missing in {experiment_name}")
        return [], []
        
    events = ea.Scalars(metric_to_plot)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

for label, folder_name in experiments.items():
    steps, values = get_data(folder_name)
    if len(steps) > 0:
        plt.plot(steps, values, label=label, linewidth=2)

plt.title("Locomotion Training: Blind vs. Privileged Critic", fontsize=16)
plt.xlabel("Training Iterations", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("comparison_result.png", dpi=300)
print("\n✅ Success! Graph saved as 'comparison_result.png'")

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# We will look at the Baseline file to find the keys
log_root = "logs/baseline_test"

# Find the file automatically
subdirs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
latest_run = sorted(subdirs)[-1]
full_path = os.path.join(log_root, latest_run)
event_file = [f for f in os.listdir(full_path) if "tfevents" in f][0]
path = os.path.join(full_path, event_file)

print(f"Inspecting: {path}")

# Load and print keys
ea = EventAccumulator(path)
ea.Reload()
print("\nAVAILABLE METRICS:")
for key in ea.Tags()['scalars']:
    print(f"  - {key}")

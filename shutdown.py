import subprocess

# Define the RL environment ('mcar', 'walker', 'l2r')
RL_env = input("Select RL environment (mcar/walker/l2r): ").strip()

# Define the training paradigm ('sequential', 'dCollect', 'dUpdate')
training_paradigm = input("Select training paradigm (sequential/dCollect/dUpdate): ").strip()

# Sanity check
assert RL_env in ("mcar", "walker", "l2r")
assert training_paradigm in ("sequential", "dCollect", "dUpdate")

# Print the info
print("---")
print(f"--- Shutting down [{RL_env}] with training paradigm [{training_paradigm}] ---")
print("---")

# Shutdown kubernetes resources
if training_paradigm == "sequential":
    # Delete the only pod
    subprocess.run(["kubectl", "delete", "pod", f"{RL_env}-sequential"], check=True)
else:
    # NOTE: change the second letter in training_paradigm to lowercase for compatibility with Kubernetes YAML
    training_paradigm = training_paradigm.lower()

    # Delete replicaset
    try:
        subprocess.run(["kubectl", "delete", "replicaset", f"{RL_env}-{training_paradigm}-workers"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Shutdown error: {e}")
    
    # Delete service
    try:
        subprocess.run(["kubectl", "delete", "service", f"{RL_env}-{training_paradigm}-learner"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Shutdown error: {e}")

    # Delete learner pod
    try:
        subprocess.run(["kubectl", "delete", "pod", f"{RL_env}-{training_paradigm}-learner"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Shutdown error: {e}")

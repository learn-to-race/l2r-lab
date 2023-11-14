import subprocess
import yaml

############################
# - Input Configurations - #
############################
# Define the RL environment ('mcar', 'walker', 'l2r')
RL_env = input("Select RL environment (mcar/walker/l2r): ").strip()

# Define the training paradigm ('sequential', 'dCollect', 'dUpdate')
training_paradigm = input("Select training paradigm (sequential/dCollect/dUpdate): ").strip()

# Define number of workers (distributed only)
num_workers = "NA"
if training_paradigm != "sequential":
    num_workers = int(input("Input number of distributed workers: ").strip())

# Define experiment name to be logged on wandb
exp_name = input("Input WandB experiment name: ").strip()

# Sanity check
assert RL_env in ("mcar", "walker", "l2r")
assert training_paradigm in ("sequential", "dCollect", "dUpdate")

# Fetch the corresponding source file
if training_paradigm == "sequential":
    source_file = "./kubernetes/template-sequential.yaml"

    with open(source_file, "r") as file:
        data = yaml.safe_load(file)
else:
    source_file = "./kubernetes/template-distributed.yaml"

    # NOTE: for distributed YAML, there are multiple sections
    stream = open(source_file, "r")
    data = yaml.load_all(stream, yaml.FullLoader)

#############################
# - Output Configurations - #
#############################
print("----------")
print(f"RL Environment = [{RL_env}] | Training Paradigm = [{training_paradigm}] | Number of Workers = [{num_workers}] | Experiment Name = [{exp_name}] ---")
print("----------")


######################################
# - Configure Template: Sequential - #
######################################
if training_paradigm == "sequential":
    # Configure metadata names (i.e., fill in the TODOs in the template YAML)
    data["metadata"]["name"] = f"{RL_env}-sequential"
    data["metadata"]["labels"]["tier"] = f"{RL_env}-sequential"
    data["spec"]["containers"][0]["name"] = f"{RL_env}-sequential"

    # Configure command
    command = data["spec"]["containers"][0]["command"][2]
    
    # NOTE: for L2R, we need to add commands to auto-launch Arrival simulator
    if RL_env == "l2r":
        command += "cd /home/LinuxNoEditor/ && sudo -u ubuntu ./ArrivalSim.sh -OpenGL & "
        command += "sleep 13m && "  # Wait for the installations to complete and the simulator to start running
        command += "source /root/anaconda3/bin/activate && conda activate initialization && source /root/.bashrc && cd /workspace/l2r-distributed && "
        command += "mamba activate l2r && git checkout sequential && "
    
    command += f"python -m scripts.main --env {RL_env} "
    command += "--wandb_apikey 173e38ab5f2f2d96c260f57c989b4d068b64fb8a "
    command += f"--exp_name {exp_name}"
    
    assert "TODO" not in str(command)
    data["spec"]["containers"][0]["command"][2] = command

#######################################
# - Configure Template: Distributed - #
#######################################
else:
    updated_yaml = []

    for idx, section in enumerate(data):
        if idx == 0:
            ##########################
            # ReplicaSet for workers #
            ##########################

            # Configure names
            worker_name = f"{RL_env}-{training_paradigm.lower()}-workers"

            section["metadata"]["name"] = worker_name
            section["metadata"]["labels"]["tier"] = worker_name
            section["spec"]["replicas"] = num_workers
            section["spec"]["selector"]["matchLabels"]["tier"] = worker_name
            section["spec"]["template"]["metadata"]["labels"]["tier"] = worker_name
            section["spec"]["template"]["spec"]["containers"][0]["name"] = worker_name

            # Configure command
            command = section["spec"]["template"]["spec"]["containers"][0]["command"][2]
            
            # NOTE: for l2r we need to start Arrival Simulator
            if RL_env == "l2r":
                command += "cd /home/LinuxNoEditor/ && sudo -u ubuntu ./ArrivalSim.sh -OpenGL & "
                command += "sleep 13m && "  # Wait for the installations to complete and the simulator to start running
                command += "source /root/anaconda3/bin/activate && conda activate initialization && source /root/.bashrc && cd /workspace/l2r-distributed && "
                command += "mamba activate l2r && "
            
            command += f" python worker.py --env {RL_env} --paradigm {training_paradigm}"
            
            section["spec"]["template"]["spec"]["containers"][0]["command"][2] = command

        elif idx == 1:
            ###################
            # Pod for learner #
            ###################

            # Configure names
            learner_name = f"{RL_env}-{training_paradigm.lower()}-learner"
            port_name = f"{RL_env}-{training_paradigm.lower()}"

            section["metadata"]["name"] = learner_name
            section["spec"]["containers"][0]["name"] = learner_name
            section["spec"]["containers"][0]["ports"][0]["name"] = port_name

            # Configure command
            command = section["spec"]["containers"][0]["command"][2]

            command += f" python server.py --env {RL_env} --paradigm {training_paradigm} --wandb_apikey 173e38ab5f2f2d96c260f57c989b4d068b64fb8a --exp_name {exp_name}"

            section["spec"]["containers"][0]["command"][2] = command
        else:
            ############################################
            # Service for worker-learner communication #
            ############################################
            section["metadata"]["name"] = f"{RL_env}-{training_paradigm.lower()}-learner"
            section["spec"]["ports"][0]["name"] = f"{RL_env}-{training_paradigm.lower()}"
            section["spec"]["ports"][0]["targetPort"] = f"{RL_env}-{training_paradigm.lower()}"

        assert "TODO" not in str(section)
        updated_yaml.append(section)

# Save the newly created kubernetes file
with open(f"./kubernetes/{RL_env}-{training_paradigm}.yaml", "w") as file:
    if training_paradigm == "sequential":
        yaml.dump(data, file)
    else:
        yaml.safe_dump_all(updated_yaml, file, default_flow_style=False)

# Launch Kubernetes source file
subprocess.run(["kubectl", "create", "-f", f"./kubernetes/{RL_env}-{training_paradigm}.yaml"], check=True)

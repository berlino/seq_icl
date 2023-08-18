import os
import yaml
import itertools
import torch
import time
from subprocess import Popen


def get_config(file):
    with open(file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def create_flags(options):
    flags = ""
    for key, value in options:
        flags += f"{key}={value} "
    return flags.strip()


def get_sweep(config):
    parameters = {}
    for key, values in config["parameters"].items():
        for value in values["values"]:
            if key not in parameters:
                parameters[key] = []
            parameters[key].append((key, value))
    flags = list(parameters.values())
    return [create_flags(options) for options in itertools.product(*flags)]


def is_free(gpu_id):
    # we want to make sure the free memory is above >20GB
    # so we don't run out of memory during training
    try:
        free_mem = torch.cuda.get_device_properties(
            gpu_id
        ).total_memory - torch.cuda.memory_allocated(gpu_id)
        return free_mem > 20e9
    except:
        print(f"GPU {gpu_id} is not available")
        return False


if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = "0"

    config = get_config("sweeps/hyper.yaml")
    sweeps = get_sweep(get_config("sweeps/hyper.yaml"))
    sweep_folder = "hyper/"
    os.makedirs(sweep_folder, exist_ok=True)

    gpus = {id: None for id in range(10)}
    print(len(sweeps))
    for i, sweep in enumerate(sweeps):
        submitted = False
        while not submitted:
            for gpu_id in gpus.keys():
                if is_free(gpu_id):
                    if gpus[gpu_id] is not None:
                        sweep_hash = gpus[gpu_id]
                        if os.path.exists(f"hyper/{sweep_hash}.done"):
                            gpus[gpu_id] = None

                    if gpus[gpu_id] is None:
                        sweep_hash = hash(sweep)
                        gpus[gpu_id] = sweep_hash
                        print(f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py {sweep}")
                        proc = Popen([
                                f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py {sweep} 1>"
                                f" hyper/{sweep_hash}.log 2> hyper/{sweep_hash}.err; touch"
                                f" hyper/{sweep_hash}.done"], shell=True)
                        submitted = True
                        break
                else:
                    print(f"GPU {gpu_id} is not free")
            if not submitted:
                time.sleep(10)

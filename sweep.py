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
    istransformer = False
    for key, value in options:
        if key == "experiment" and "transformer" in value:
            istransformer = True

    for key, value in options:
        flags += f"{key}={value} "
        if istransformer and "model.n_layer" in key:
            attn_layer_idx = str(list(range(value)))
            flags += f"model.attn_layer_idx=\"{attn_layer_idx}\" "
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
    sweep_folder = "hyper/fixes/hyper"
    os.makedirs(sweep_folder, exist_ok=True)
    hash_offset = 128

    gpus = {id: None for id in [2, 3, 4, 5, 10, 11, 12, 13, 14, 15]}
    print(len(sweeps))
    for i, sweep in enumerate(sweeps):
        submitted = False
        while not submitted:
            for gpu_id in gpus.keys():
                if is_free(gpu_id):
                    if gpus[gpu_id] is not None:
                        sweep_hash = gpus[gpu_id]
                        if os.path.exists(f"{sweep_folder}/{sweep_hash}.done"):
                            gpus[gpu_id] = None

                    if gpus[gpu_id] is None:
                        sweep_hash = hash(sweep) + hash_offset
                        gpus[gpu_id] = sweep_hash
                        command = (
                            f"export CUDA_VISIBLE_DEVICES={gpu_id}; python -c "
                            f'"import pykeops; pykeops.clean_pykeops();"; python train.py wandb.project="fixes-hyper" {sweep} 1> {sweep_folder}/{sweep_hash}.log 2> {sweep_folder}/{sweep_hash}.err; touch {sweep_folder}/{sweep_hash}.done'
                        )
                        print(command)
                        proc = Popen(
                            [command],
                            shell=True,
                        )
                        submitted = True
                        break
                else:
                    pass
                    # print(f"GPU {gpu_id} is not free")
            if not submitted:
                time.sleep(10)

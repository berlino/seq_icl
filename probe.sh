export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=0; python probe.py --layer 0 --use_wandb --use_ratio  > exps/probe/l0 2> exps/probe/l0.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=1; python probe.py --layer 1 --use_wandb --use_ratio  > exps/probe/l1 2> exps/probe/l1.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=2; python probe.py --layer 2 --use_wandb --use_ratio  > exps/probe/l2 2> exps/probe/l2.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=3; python probe.py --layer 3 --use_wandb --use_ratio  > exps/probe/l3 2> exps/probe/l3.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=4; python probe.py --layer 4 --use_wandb --use_ratio  > exps/probe/l4 2> exps/probe/l4.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=5; python probe.py --layer 5 --use_wandb --use_ratio   > exps/probe/l5 2> exps/probe/l5.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=6; python probe.py --layer 6 --use_wandb --use_ratio   > exps/probe/l6 2> exps/probe/l6.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=7; python probe.py --layer 7 --use_wandb --use_ratio   > exps/probe/l7 2> exps/probe/l7.err &


export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=8; python probe.py --layer 0 --use_wandb  > exps/probe/l0c 2> exps/probe/l0c.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=9; python probe.py --layer 1 --use_wandb   > exps/probe/l1c 2> exps/probe/l1c.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=10; python probe.py --layer 2 --use_wandb  > exps/probe/l2c 2> exps/probe/l2c.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=11; python probe.py --layer 3 --use_wandb  > exps/probe/l3c 2> exps/probe/l3c.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=12; python probe.py --layer 4 --use_wandb  > exps/probe/l4c 2> exps/probe/l4c.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=13; python probe.py --layer 5 --use_wandb    > exps/probe/l5c 2> exps/probe/l5c.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=14; python probe.py --layer 6 --use_wandb   > exps/probe/l6c 2> exps/probe/l6c.err &
export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=15; python probe.py --layer 7 --use_wandb   > exps/probe/l7c 2> exps/probe/l7c.err &

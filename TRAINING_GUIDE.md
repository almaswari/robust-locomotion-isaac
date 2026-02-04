# legged_gym Training Guide

## Quick Start


### Update and upgrade
```bash

```
### Train ANYmal C on Flat Terrain
```bash
cd ~/legged_gym
./train.sh --task=anymal_c_flat --num_envs=256 --headless
```

### Train with Visualization
```bash
./train.sh --task=anymal_c_flat --num_envs=256
```

### Available Tasks
- `anymal_c_flat` - ANYmal C on flat terrain (easiest to train)
- `anymal_c_rough` - ANYmal C on rough terrain
- `anymal_b` - ANYmal B robot
- `a1` - Unitree A1 robot
- `cassie` - Cassie bipedal robot

### Training Parameters
- `--task` - Robot and environment to train
- `--num_envs` - Number of parallel environments (default: 4096)
- `--headless` - Run without visualization (faster)
- `--max_iterations` - Training iterations (default: 300)

## Monitor Training

### TensorBoard
Training logs are saved to `~/legged_gym/logs/`. View them with:
```bash
conda activate rl_loco
cd ~/legged_gym
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Environment Details

- **Conda Environment**: `rl_loco`
- **Python**: 3.8.20
- **GPU**: Quadro P1000
- **CUDA**: 12.8
- **System**: WSL2

## Troubleshooting

If you get CUDA errors, make sure the launch script includes:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ky_ode/miniconda3/envs/rl_loco/lib:/usr/lib/wsl/lib
```

This is required for WSL to find CUDA libraries.

to run the tranied model 
```bash
python legged_gym/scripts/play.py --task=a1 --num_envs=1 --load_run Feb03_04-14-39_ 
```
```bash
--load_run Feb03_00-38-34_ # this the model name
```




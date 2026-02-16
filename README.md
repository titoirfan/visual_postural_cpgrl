# CPG-RL for Quadrupedal Locomotion, with Exteroception and Postural Adaptation

Supporting code and files for "Synergy of Postural Adaptation and Exteroception for Robust CPG-Driven Quadrupedal Locomotion".


## Links

- Trained models evaluated in the paper are available [here](https://drive.google.com/file/d/1DhYE_RvU2ClxBJzo0IdZwbjsDN6mu0fX/view?usp=sharing).
- Source data for reproducing the figures and tables depicted in the paper are available [here](https://drive.google.com/file/d/16_xPgjz9dNGrugBWv9cfmxOzfvYlMGcD/view?usp=sharing).


---

## Dependencies
- IsaacSim v4.5.0
- Isaac Lab v2.1.0


## Installation

- Install Isaac Lab by following the installation guide in the [official documentations](https://isaac-sim.github.io/IsaacLab/main/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.
- Clone this repository separately from the Isaac Lab installation.
- Using a python interpreter that has Isaac Lab installed, install the library
```bash
python -m pip install -e exts/cpg
```


## Using this code

You can check the list of available environments in `exts/cpg/cpg/tasks/cpg/__init__.py`.

To train the CPG policy, invoke:
```
python scripts/rsl_rl/train.py --task CPG-Rough-Unitree-A1-v0 --num_envs <int num_envs> --headless --run_name <string run_name> --seed <int seed>
```
You can then play the policy using the `play.py` script and the corresponding play environment.

To enable exteroception and postural reflex, you can modify `enable_exteroception` and `enable_reflex_network` in `exts/cpg/cpg/tasks/cpg/cpg_env_cfg.py`.

You can also evaluate the policy's performance using the `eval.py` script, which will record logs and calculate performance metrics. 
See `run_evals.sh` for example on how to use the script.
The recorded logs can be processed using scripts in `scripts/eval`.


## Citing this work

To be added.

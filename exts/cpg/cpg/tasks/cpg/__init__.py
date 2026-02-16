# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Anymal-C locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Direct joint position control environments
gym.register(
    id="Direct-Flat-Unitree-A1-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_a1_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Direct-Rough-Unitree-A1-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_a1_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Direct-Flat-Unitree-A1-Play-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_a1_env_cfg:UnitreeA1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Direct-Rough-Unitree-A1-Play-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_a1_env_cfg:UnitreeA1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Direct-Benchmark-Unitree-A1-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.benchmark_env_cfg:UnitreeA1BenchmarkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1BenchmarkPPORunnerCfg",
    },
)

# CPG environments
gym.register(
    id="CPG-Flat-Unitree-A1-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1FlatPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Stairs-Unitree-A1-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1StairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1StairsPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Flat-Unitree-A1-Play-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1FlatPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Play-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Stairs-Unitree-A1-Play-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1StairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1StairsPPORunnerCfg",
    },
)

# Evaluation environments
gym.register(
    id="CPG-Rough-Unitree-A1-Eval-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Rough-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_IdealRough",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Discrete-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_IdealDiscrete",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Wave-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_IdealWave",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Push-Flat-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_PushFlat",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Push-Rough-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_PushRough",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Push-Discrete-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_PushDiscrete",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Push-Wave-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_PushWave",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Gait-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_Gait",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Stability-SideS-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_StabilitySideS",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Stability-SideM-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_StabilitySideM",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Stability-SideL-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_StabilitySideL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Tracking-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_Tracking",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Tracking-Discrete-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_TrackingDiscrete",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Stress-Heavy-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_StressHeavy",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Stress-Slippery-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_StressSlippery",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Rough-Unitree-A1-Eval-Stress-Heavy-Slippery-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1RoughEnvCfg_EVAL_StressHeavySlippery",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Stairs-Unitree-A1-Eval-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1StairsEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1StairsPPORunnerCfg",
    },
)

gym.register(
    id="CPG-Stairs-Unitree-A1-Eval-Push-v0",
    entry_point=f"{__name__}.cpg_env:CPGUnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cpg_env_cfg:CPGUnitreeA1StairsEnvCfg_EVAL_Push",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CPGUnitreeA1StairsPPORunnerCfg",
    },
)

gym.register(
    id="Direct-Benchmark-Unitree-A1-Eval-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.benchmark_env_cfg:UnitreeA1BenchmarkEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1BenchmarkPPORunnerCfg",
    },
)

gym.register(
    id="Direct-Benchmark-Unitree-A1-Eval-Discrete-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.benchmark_env_cfg:UnitreeA1BenchmarkEnvCfg_EVAL_IdealDiscrete",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1BenchmarkPPORunnerCfg",
    },
)

gym.register(
    id="Direct-Benchmark-Unitree-A1-Eval-Push-Discrete-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.benchmark_env_cfg:UnitreeA1BenchmarkEnvCfg_EVAL_PushDiscrete",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1BenchmarkPPORunnerCfg",
    },
)

gym.register(
    id="Direct-Benchmark-Unitree-A1-Eval-Gait-v0",
    entry_point=f"{__name__}.unitree_a1_env:UnitreeA1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.benchmark_env_cfg:UnitreeA1BenchmarkEnvCfg_EVAL_Gait",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1BenchmarkPPORunnerCfg",
    },
)

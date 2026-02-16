import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.15), noise_step=0.01, downsampled_scale=0.2, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.35, grid_width=0.45, grid_height_range=(0.01, 0.06), platform_width=1.0
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.35, amplitude_range=(0.02, 0.1), num_waves=8, border_width=0.25
        ),
    },
)

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.02, 0.10), step_width=0.3, platform_width=2.0, border_width=0.5, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.02, 0.10), step_width=0.3, platform_width=2.0, border_width=0.5, holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.06), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
    },
)

EVAL_FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(40.0, 40.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    curriculum = False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

EVAL_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(40.0, 40.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    curriculum = False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.01, 0.15), noise_step=0.01, downsampled_scale=0.2, border_width=0.0
        ),
    },
)

EVAL_DISCRETE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(40.0, 40.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    curriculum = False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.01, 0.06), platform_width=1.0
        ),
    },
)

EVAL_WAVE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(40.0, 40.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    curriculum = False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0, amplitude_range=(0.02, 0.1), num_waves=40, border_width=0.0
        ),
    },
)

EVAL_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    curriculum = False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5, step_height_range=(0.02, 0.10), step_width=0.3, platform_width=2.0, border_width=0.5, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5, step_height_range=(0.02, 0.10), step_width=0.3, platform_width=2.0, border_width=0.5, holes=False,
        ),
    },
)

from .utils import choose_root


CC359_DATA_PATH = choose_root(
    '/nmnt/x3-hdd/data/da_mri/cc359',
    '/gpfs/data/gpfs0/b.shirokikh/data/cc359',
)

CC359_BASELINE_PATH = choose_root(
    '/gpfs/data/gpfs0/b.shirokikh/experiments/da/miccai2021_spottune/baseline/cc359_unet2d_one2all/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

WMH_UNET3D_BASELINE_PATH = choose_root(
    '/gpfs/data/gpfs0/b.shirokikh/experiments/da/spottune/wmh/baseline/wmh_unet3d_one2all/',
    '/nmnt/x4-hdd/experiments/da_exps/wmh/baseline/wmh_unet3d_one2all/',
)

WMH_DM_BASELINE_PATH = choose_root(
    '/gpfs/data/gpfs0/b.shirokikh/experiments/da/spottune/wmh/baseline/wmh_dm_one2all/',
    '/nmnt/x4-hdd/experiments/da_exps/wmh/baseline/wmh_dm_one2all/',
)

WMH_DATA_PATH = choose_root(
    '/nmnt/x3-hdd/data/da_mri/wmh_ants',
    '/gpfs/data/gpfs0/b.shirokikh/data/wmh_ants',
)

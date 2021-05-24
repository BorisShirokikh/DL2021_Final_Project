from .utils import choose_root


CC359_DATA_PATH = choose_root(
    '/nmnt/x3-hdd/data/da_mri/cc359',
    '/gpfs/data/gpfs0/b.shirokikh/data/cc359',
    '/',  # TODO: avoiding `FileNotFoundError`
)

CC359_BASELINE_PATH = choose_root(
    '/gpfs/data/gpfs0/b.shirokikh/experiments/da/miccai2021_spottune/baseline/cc359_unet2d_one2all/',
    '/',  # TODO: avoiding `FileNotFoundError`
)

WMH_DATA_PATH = choose_root(
    '/nmnt/x3-hdd/data/da_mri/wmh_ants',
    '/gpfs/data/gpfs0/b.shirokikh/data/wmh_ants',
    '/nmnt/media/home/anastasia_kurmukova/DL2021_Final_Project/wmh_ants',  # TODO: avoiding `FileNotFoundError`
)

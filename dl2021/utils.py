import os
import random
from pathlib import Path

import numpy as np
import torch
import surface_distance.metrics as surf_dc

from dpipe.io import PathLike
from dpipe.im.axes import broadcast_to_axis, AxesLike, AxesParams, axis_from_dim, resolve_deprecation
from dpipe.im.grid import crop_to_box, combine,get_boxes
from dpipe.itertools import extract, pmap
from dpipe.im.shape_ops import pad_to_shape, crop_to_shape, pad_to_divisible
from dpipe.im.shape_utils import prepend_dims, extract_dims
from dpipe.im.shape_ops import pad
from typing import Iterable, Union, Callable

from functools import wraps


def get_pred(x, threshold=0.5):
    return x > threshold


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_root(*paths: PathLike) -> Path:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError('No appropriate root found.')


def sdice(a, b, spacing, tolerance):
    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)


def skip_predict(output_path):
    print(f'>>> Passing the step of saving predictions into `{output_path}`', flush=True)
    os.makedirs(output_path)

def divide_grid(x: np.ndarray, max_axes: np.ndarray, patch_size: AxesLike, stride: AxesLike, axis: AxesLike = None,
           valid: bool = False) -> Iterable[np.ndarray]:
    """
    A convolution-like approach to generating patches from a tensor.
    Parameters
    ----------
    x
    patch_size
    axis
        dimensions along which the slices will be taken.
    stride
        the stride (step-size) of the slice.
    valid
        whether patches of size smaller than ``patch_size`` should be left out.
    References
    ----------
    See the :doc:`tutorials/patches` tutorial for more details.
    """
    min_axes = np.array([0]*3)
    # max_axes = np.array([198, 330, 144])
    for box in get_boxes(x.shape, patch_size, stride, axis, valid=valid):
        # print(box)
        padding = np.array([[0] * 2] * 3)
        first_axis = np.array(box[0,2:]) - np.array([24,24,24]) - min_axes
        second_axis = max_axes - np.array(box[1,2:]) - np.array([24,24,24])
        # print(first_axis, second_axis)
        padding[:, 0][first_axis < 0] = first_axis[first_axis < 0]*-1
        padding[:, 1][second_axis < 0] = second_axis[second_axis < 0]*-1
        box1 = np.concatenate((box[0,:2], np.max((min_axes, np.array(box[0,2:]) - np.array([24,24,24])), axis=0)))
        box2 = np.concatenate((box[1,:2], np.min((max_axes, np.array(box[1,2:]) + np.array([24,24,24])), axis=0)))
        new_box = np.vstack((box1,box2))
        # print(new_box, padding)
        # print((max_axes, np.array(box[1,2:]) + np.array([24,24,24])), new_box_upp)
        ans = pad(crop_to_box(x, new_box), padding=padding, padding_values=np.min, axis=axis)
        # print("ANS ", ans.shape)
        yield ans

def patches_grid_dm(patch_size: AxesLike, stride: AxesLike, axis: AxesLike = None,
                 padding_values: Union[AxesParams, Callable] = 0, ratio: AxesParams = 0.5):
    """
    Divide an incoming array into patches of corresponding ``patch_size`` and ``stride`` and then combine
    predicted patches by averaging the overlapping regions.
    If ``padding_values`` is not None, the array will be padded to an appropriate shape to make a valid division.
    Afterwards the padding is removed.
    References
    ----------
    `grid.divide`, `grid.combine`, `pad_to_shape`
    """
    valid = padding_values is not None

    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            input_axis = resolve_deprecation(axis, x.ndim, patch_size, stride)
            local_size, local_stride = broadcast_to_axis(input_axis, patch_size, stride)

            if valid:
                shape = extract(x.shape, input_axis)
                padded_shape = np.maximum(shape, local_size)
                new_shape = padded_shape + (local_stride - padded_shape + local_size) % local_stride
                x = pad_to_shape(x, new_shape, input_axis, padding_values, ratio)

            patches = pmap(predict, divide_grid(x, new_shape, local_size, local_stride, input_axis), *args, **kwargs)
            prediction = combine(patches, extract(x.shape, input_axis), local_stride, axis)

            if valid:
                prediction = crop_to_shape(prediction, shape, axis, ratio)
            return prediction

        return wrapper

    return decorator


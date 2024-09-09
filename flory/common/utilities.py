"""Utilities shared by package :mod:`flory`.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import inspect
from typing import Any, Callable
import numpy as np


def filter_kwargs(kwargs_full: dict[str, Any], func: Callable) -> dict[str, Any]:
    """Filter the keyword arguments (dict) not accepted by a function.

    Args:
        kwargs_full:
            The dictionary for all keyword arguments, including the redundant ones.
        func:
            The function to check against.

    Returns:
        : The filtered dictionary
    """
    params = inspect.signature(func).parameters.keys()
    return {para: kwargs_full[para] for para in params if para in kwargs_full}

def convert_and_broadcast(arr: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Converts input and broadcasts it to an array with specified shape.

    Args:
        arr: 
            The input array.
        shape: 
            The target shape to broadcast to.

    Returns:
        : The broadcasted array.
    """
    ans = np.atleast_1d(arr)
    ans = np.array(np.broadcast_to(ans, shape))
    return ans

def make_square_blocks(arr: np.ndarray, block_sizes: np.ndarray) -> np.ndarray:
    """Expands a square np.ndarray into blocks.

    Args:
        arr: 
            The input array. Must be square (i.e., all dimensions are equal).
        block_sizes: 
            The sizes for the blocks. Must have the same length as :paramref:`arr`.

    Returns:
        : The input array expanded into blocks. Each element of :paramref:`arr` is
        repeated according to :paramref:`block_sizes` to create blocks.
    """
    arr_shape = arr.shape
    n_block = block_sizes.shape[0]

    if len(set(arr_shape)) != 1 or n_block != arr_shape[0]:
        raise ValueError(
            "The array to be extended to blocks must be square and match the number of blocks."
        )

    ans = arr
    for id in range(len(arr_shape)):
        ans = np.repeat(ans, block_sizes, axis=id)

    return ans

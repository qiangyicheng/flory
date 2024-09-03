"""Utilities shared by package :mod:`flory`.

.. codeauthor:: Yicheng Qiang <yicheng.qiang@ds.mpg.de>
"""

import inspect
from typing import Any, Callable


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

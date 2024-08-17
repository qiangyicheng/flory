import inspect
from typing import Callable

def filter_kwargs(kwargs_full: dict["str", any], func: Callable) -> dict["str", any]:
    """Filter the keyword arguments (dict) accepted by a function

    Args:
        kwargs_full:
            The dictionary for all keyword arguments, including the redundant ones.
        func:
            The function to check against

    Returns:
        : The filtered dictionary
    """
    params = inspect.signature(func).parameters.keys()
    return {para: kwargs_full[para] for para in params if para in kwargs_full}


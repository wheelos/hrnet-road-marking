"""Small replacements for upstream runx and Apex interpolation decorators."""
from contextlib import nullcontext
import functools
import logging
from types import SimpleNamespace

import torch


def float_function(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        floating = [a for a in args if isinstance(a, torch.Tensor) and a.is_floating_point()]
        context = (torch.autocast('cuda', enabled=False)
                   if any(a.is_cuda for a in floating) else nullcontext())
        with context:
            args = tuple(a.float() if isinstance(a, torch.Tensor) and a.is_floating_point()
                         else a for a in args)
            return function(*args, **kwargs)
    return wrapped


amp = SimpleNamespace(float_function=float_function)
logx = SimpleNamespace(msg=lambda message, *args, **kwargs: logging.info(message))


def fmt_scale(prefix, scale):
    return f'{prefix}_{float(scale)}x'

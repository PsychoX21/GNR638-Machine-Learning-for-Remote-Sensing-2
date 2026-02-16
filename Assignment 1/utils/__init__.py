"""Utility package for DeepNet framework"""

from .metrics import count_parameters, estimate_macs_flops
__all__ = [
    'count_parameters',
    'estimate_macs_flops',
]


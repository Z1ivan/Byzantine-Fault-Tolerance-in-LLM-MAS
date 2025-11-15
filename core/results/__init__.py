#!/usr/bin/env python3

from .result_processor import StandardizedResultProcessor

__version__ = "2.1.0"
__author__ = "Byzantine Project Team"

__all__ = [
    'StandardizedResultProcessor',

    '__version__',
    '__author__'
]

def create_standard_processor(output_dir: str):

    return StandardizedResultProcessor(output_dir)

def get_supported_methods():
    return ['gsm8k', 'safe', 'prompt_probe']

def validate_method_type(method_type: str) -> bool:

    return method_type in get_supported_methods()

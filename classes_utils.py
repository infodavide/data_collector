# -*- coding: utf-8 -*-
"""
Utilities for classes
"""
import importlib
import os
import sys


def subclasses_of(cls) -> set[type]:
    """
    Return the subclasses of the given class
    :param cls: the base class
    :return: the subclasses
    """
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in subclasses_of(c)])


def import_files_of_dir(path: str) -> None:
    """
    Import .py files of the given directory
    :param path: the directory
    """
    __globals = globals()
    if not os.path.isdir(path):
        return
    sys.path.append(path)
    for file in os.listdir(path):
        if file.startswith('_') or not file.lower().endswith('.py'):
            continue
        mod_name = file[:-3]  # strip .py at the end
        if mod_name not in __globals:
            __globals[mod_name] = importlib.import_module(mod_name)
            __import__(mod_name)

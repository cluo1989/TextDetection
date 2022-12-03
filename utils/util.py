'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-11-30 10:55:19
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-11-30 10:55:26
FilePath: /dbnet_plus/modeling/architectures/util.py
Description: common tools
'''
# coding: utf-8
import logging
import importlib


def initial_logger():
    FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger


def create_module(module_str):
    tmpss = module_str.split(",")
    assert len(tmpss) == 2, "Error formate\
        of the module path: {}".format(module_str)
    module_name, function_name = tmpss[0], tmpss[1]
    somemodule = importlib.import_module(module_name, __package__)
    function = getattr(somemodule, function_name)
    return function
    
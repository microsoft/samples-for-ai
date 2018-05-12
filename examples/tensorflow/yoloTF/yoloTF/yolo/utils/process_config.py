#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:09:41 2018

@author: root
"""

import configparser

def processConfig(configFile):
    '''
    Process configure file.
    
    Args:
        configFile: path to the configure file
        
    Return:
        CommonParams, datasetParams, netParams, solverParams
    '''
    commonParams = {}
    datasetParams = {}
    netParams = {}
    solverParams = {}
    
    config = configparser.ConfigParser()
    config.read(configFile)
    
    for section in config.sections():
        if section == 'Common':
            for option in config.options(section):
                commonParams[option] = config.get(section, option)
                
        if section == 'DataSet':
            for option in config.options(section):
                datasetParams[option] = config.get(section, option)
                
        if section == 'Net':
            for option in config.options(section):
                netParams[option] = config.get(section, option)
        if section == 'Solver':
            for option in config.options(section):
                solverParams[option] = config.get(section, option)
    
    return commonParams, datasetParams, netParams, solverParams
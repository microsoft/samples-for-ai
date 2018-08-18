#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:53:42 2018

@author: zsc
"""

import sys
from optparse import OptionParser

sys.path.append('./')

import yolo
from yolo.utils.process_config import processConfig

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure", help="configure filename")

(option, args) = parser.parse_args()

if option.configure:
    confFile = str(option.configure)
    
else:
    print('error file')
    exit(0)

commonParams, datasetParams, netParams, solverParams = processConfig(confFile)
# yolo.text_dataset.text_dataset.TextDataSet
dataset = eval(datasetParams['name'])(commonParams, datasetParams)
# yolo.net.yolo_tiny_net.YoloTinyNet
net = eval(netParams['name'])(commonParams, netParams)
#yolo.yolo_solver.yolo_solver.YoloSolver
solver = eval(solverParams['name'])(dataset, net, commonParams, solverParams)
solver.solve()
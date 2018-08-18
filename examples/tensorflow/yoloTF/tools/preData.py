#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:26:39 2018

@author: zsc
"""

import os
import xml.etree.ElementTree as ET
import struct
import numpy as np

classesName =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


classesNum = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19}

"""
Define the yoloPath, dataPath and output path.
"""
yoloRoot = os.path.abspath('./')
dataPath = os.path.join(yoloRoot, 'data/VOCdevkit2007')
outputPath = os.path.join(yoloRoot, 'data/pascal_voc.txt')

def parseXML(xmlFile):
    """
    Parse specified xml file.
    
    Args:
        xmlFile: the xml file path
        
    Returns:
        imagePath: a string to identify the image path.
        labels: list of [xmin, ymin, xmax, ymax, class]
    """
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    imagePath = ''
    labels = []
    
    for item in root:
        if item.tag == 'filename':
            imagePath = os.path.join(dataPath, 'VOC2007/JPEGImages', item.text)
        elif item.tag == 'object':
            objName = item[0].text
            objNum = classesNum[objName]
            xmin = int(item[4][0].text)
            ymin = int(item[4][1].text)
            xmax = int(item[4][2].text)
            ymax = int(item[4][3].text)
            labels.append([xmin, ymin, xmax, ymax, objNum])
    return imagePath, labels

def convertXML(imagePath, labels):
    """
    Convert imagePath, labels to string.
    
    Args:
        imagePath
        labels
        
    Return:
        outString: a string of image path & labels.
    """
    outString = ''
    outString += imagePath
    for label in labels:
        for i in label:
            outString += ' ' + str(i)
    outString += '\n'
    return outString

def main():
    
    outFile = open(outputPath, 'w')
    xmlDir = dataPath + '/VOC2007/Annotations/'
    
    xmlList = os.listdir(xmlDir)
    xmlList = [xmlDir + temp for temp in xmlList]
    
    for xml in xmlList:
        try:
            imagePath, labels = parseXML(xml)
            record = convertXML(imagePath, labels)
            outFile.write(record)
        except Exception:
            pass
    outFile.close()
    

if __name__ == '__main__':
    main()
            
            

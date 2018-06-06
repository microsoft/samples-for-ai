"""
Modification of https://raw.githubusercontent.com/carpedm20/DCGAN-tensorflow/master/download.py

Downloads the following:
- Celeb-A dataset
"""

from __future__ import print_function
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import requests
import subprocess
from tqdm import tqdm
from six.moves import urllib




def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=1024 * 1024):
    total_size = int(response.headers.get('content-length', 1476619107))
    #print(response.headers)
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size, unit = 'MB',  unit_scale=True, 
                          #desc=destination
                          ):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_file_from_github(save_path):
    os.system('git clone https://github.com/Somedaywilldo/celeba_dataset & copy ./celeba_dataset '+save_path)


def download_celeb_a(dirpath):
    try:        
        prepare_data_dir(dirpath)
        data_dir = 'CelebA'  
        if os.path.exists(os.path.join(dirpath, data_dir)):
            print('Found Celeb-A - skip')
            return os.path.join(dirpath, data_dir)

        filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
        save_path = os.path.join(dirpath, filename)

        if os.path.exists(save_path):
            print('[*] {} already exists'.format(save_path))
        else:
            download_file_from_google_drive(drive_id, save_path)
        #download_file_from_github(save_path)

    

        zip_dir = ''
        with zipfile.ZipFile(save_path) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(dirpath)
        os.remove(save_path)
        os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))
        return os.path.join(dirpath, data_dir)
    except:
        shutil.rmtree(dirpath)
        print("\nDownload failed, please retry! Or get datasets from Baidu Netdisk following the instructions in README.")
        exit(0)

def prepare_data_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    download_celeb_a('datasets')

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:23:27 2021

@author: conor
"""

import os
os.chdir(r"C:\Users\conor\Documents\FYP\LoVe\TEST")
import pandas as pd
import ketos.neural_networks.dev_utils.detection as det
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.resnet import ResNetInterface
from ketos.neural_networks.dev_utils.detection import process, save_detections
import matplotlib.pyplot as plt
import glob
import csv
#%matplotlib inline
#import packages
model, audio_repr = ResNetInterface.load_model_file(model_file='FW20.kt', new_model_folder='./FW20_tmp_folder', load_audio_repr=True)
#load the classifier
annot_test = pd.read_csv('annot_test.csv')
spec_config = audio_repr[0]['spectrogram']
#load annotations file and spectrogram settings
file = glob.glob('./test_data/*')
x=[]
for path in file:
    print(path)
    audio_loader = AudioFrameLoader(step=0.5, filename=path, repres=spec_config)
    det = process(audio_loader, model=model, batch_size=64, progress_bar=True, group=True, threshold=0.5, win_len=5)
    x = x + det
#detection process across all wav files, saved to variable x
save_detections(detections=x, save_to='detections.csv')
#write detections to csv file
def compare(annotations, detections):

    detected_list = []

    for idx,row in annotations.iterrows(): #loop over annotations
        filename_annot = row['sound_file']
        time_annot = row['call_time']
        detected = False
        for d in detections: #loop over detections
            filename_det = d[0]
            start_det    = d[1]
            end_det      = start_det + d[2]
            # if the filenames match and the annotated time falls with the start and 
            # end time of the detection interval, consider the call detected
            if filename_annot==filename_det and time_annot >= start_det and time_annot <= end_det:
                detected = True
                break

        detected_list.append(detected)       

    annotations['detected'] = detected_list  #add column to the annotations table

    return annotations
#define function to compare detections with annotations 
annot = compare(annot_test, x)
annot.to_csv('comparison.csv')
#call the function and write results to csv file
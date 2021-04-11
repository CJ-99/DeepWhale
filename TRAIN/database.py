# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:15:13 2021

@author: conor
"""

import os
os.chdir(r"C:\Users\conor\Documents\FYP\LoVe\TRAIN")
#change directory
import pandas as pd
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.audio.spectrogram import MagSpectrogram
from ketos.data_handling.parsing import load_audio_representation
import numpy as np
from sklearn.model_selection import train_test_split
#import packages
train_full = pd.read_csv("annot_train.csv")
annot_train, annot_val = train_test_split(train_full, test_size=0.2)
#split train data

map_to_ketos_annot_std ={'sound_file': 'filename'} 
std_annot_train = sl.standardize(table=annot_train, signal_labels=["FW20"], mapper=map_to_ketos_annot_std, trim_table=True)
std_annot_val = sl.standardize(table=annot_val, signal_labels=["FW20"], mapper=map_to_ketos_annot_std, trim_table=True)
#map to ketos format

positives_train = sl.select(annotations=std_annot_train, length=3.0)
positives_val = sl.select(annotations=std_annot_val, length=3.0)
#create positives of uniform length

file_durations_train = sl.file_duration_table('train_data')
negatives_train=sl.create_rndm_backgr_selections(annotations=std_annot_train, files=file_durations_train, length=3.0, num=len(positives_train), trim_table=True)
negatives_val=sl.create_rndm_backgr_selections(annotations=std_annot_val, files=file_durations_train, length=3.0, num=len(positives_val), trim_table=True)
#create negatives same length as positives

selections_train = positives_train.append(negatives_train, sort=False)
selections_val = positives_val.append(negatives_val, sort=False)
#append positives with negatives

spec_cfg = load_audio_representation('spec_config.json', name="spectrogram")
#load spectrogram settings

dbi.create_database(output_file='database.h5', data_dir='train_data',
                               dataset_name='train',selections=selections_train,
                               audio_repres=spec_cfg)
dbi.create_database(output_file='database.h5', data_dir='train_data',
                               dataset_name='val',selections=selections_val,
                               audio_repres=spec_cfg)
#create database


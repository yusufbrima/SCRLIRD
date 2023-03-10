#!/usr/bin/env python3
# qsub -v PATH -v PYTHONPATH=$PWD -cwd -l mem=8G,cuda=1 Playground.py
from __future__ import print_function
import argparse
import random
import warnings
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sn
import librosa
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import logging
import conf as config 
from util import Preprocessing

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

def extract_features(file):
    pp  = Preprocessing(datapath=None)
    y,sr = librosa.load(file , sr = None, mono=True)

    # mfcc =  pp.read_mfcc(y, sample_rate=sr)

    smfcc = pp.sample_from_mfcc(pp.read_mfcc(y, sample_rate=sr), config.dataparam['NUM_FRAMES']) 
    return np.expand_dims(smfcc, axis=0)

def scipy_cosine(x,y):
    return 1. - cdist(x, y, 'cosine')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    
    files  = [ Path("./Samples",  x) for x in os.listdir("./Samples") ]

    # Extracting features of speaker 1
    # f1 = Path("Samples/Speaker_0_0.wav")
    # f2 = Path("Samples/Speaker_0_1.wav")
    f1 = Path("Samples/clip13_loango_181117_11_16_26_PAN_ph_rt_ch.wav")
    f2 = Path("Samples/loa_LmS_181117_09_29_24_PAN_ph,dr_unclear_ch.wav")
    X1 = extract_features(f1)
    X2 = extract_features(f2)



    # Extracting features of speaker 2
    f3 = Path("Samples/loa_LmS_190103_14_23_55_LOU_ph_dr_ph_tv_co.wav")
    f4 = Path("Samples/loa_LmS_190109_08_32_36_LOU_ph,tb.wav")
    X3 = extract_features(f3)
    X4 = extract_features(f4)

    model_paths = "/net/store/cv/users/ybrima/RTGCompCog/ChimNet/Models/"

    # model =  tf.keras.models.load_model(Path(model_paths,'ResNet50_InfoNCE_encoder'), compile=False)
    model =  tf.keras.models.load_model(Path(model_paths,'EfficientNetB7_Baseline_Classifier_Chimp'), compile=True)
    model =  tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    print(model.summary())
    X = model(X1)
    X2 = model(X2)
    X3 = model(X3)
    X4 = model(X4)

    print(f"Sklearn, same speaker {cosine_similarity(X, X2),cosine_similarity(X3, X4)}, different speakers{cosine_similarity(X, X3),cosine_similarity(X, X4)}")

    print(f"Scipy, same speaker {scipy_cosine(X, X2),scipy_cosine(X3, X4)}, different speakers{scipy_cosine(X, X3),scipy_cosine(X, X4)}")




    
from __future__ import print_function
import random
import logging
import warnings
import os
import shutil
import glob
from pathlib import Path
import librosa.display
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import cv2
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from audiomentations import Compose,AddGaussianNoise,TimeStretch,PitchShift,Shift
from python_speech_features import fbank
import matplotlib.pyplot as plt
import splitfolders
import h5py
import conf as config
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


class Preprocessing:
    def __init__(self, datapath = "") -> None:
        if datapath != "":
            self.filepaths =  datapath
        else:
            self.filepaths = config.filepaths['data_path']
        self.datapath =  config.filepaths['file_path']
        self.basepath =  config.filepaths['base_path']
        self.waveaugment = [AddGaussianNoise(min_amplitude=0.001,max_amplitude=0.015,p=1.0),TimeStretch(min_rate=0.8,\
            max_rate=1.25,p=1.0),PitchShift(min_semitones=-4,max_semitones=4, p=1.0),Shift(min_fraction=-0.5, max_fraction=0.5,p=1.0)]
        logging.info(self.filepaths)
    def get_classes(self) -> list:
        """
          Generates a class list of directories containing audio samples
        """
        C = []
        for d in os.listdir(self.filepaths):
            temp =  Path(self.filepaths, d)
            if os.path.isdir(temp) and len(glob.glob(f"{temp}/*.wav")) > 0:
                C.append(d)
        return C
    def convert_encoding(input_dir:str, output_dir: str) -> None:
        outpath = Path(output_dir)
        outpath.mkdir(parents=True, exist_ok=True)
        for d in os.listdir(Path(input_dir)):
            current_path = Path(outpath, d)
            current_path.mkdir(parents=True, exist_ok=True)
            temp = Path(input_dir,d)
            if os.path.isdir(temp):
                for file in temp.glob("**/*.wav"):
                    filename =  Path(temp,file)
                    y,sr = sf.read(filename)
                    sf.write(Path(current_path,filename.name), y, sr, subtype='PCM_16')
    def get_files(self):
        """
           This method moves through the directories and creates a dataframe of audio files and their class string names and indicies
           returns a dataframe of dimension n_samples x 5, where 5 is the tuple of features reads
        """
        ds =  {'file': [], 'class': [], 'label':  [], 'duration': [],'sr': []}
        CLASSES =  self.get_classes()
        for d in os.listdir(self.filepaths):
            temp =  Path(self.filepaths, d)
            if os.path.isdir(temp):
                for file in temp.glob("**/*.wav"):
                    filename =  Path(temp,file)
                    y,sr = librosa.load(filename, sr=None)
                    duration =  (1/sr) * len(y)
                    ds['file'].append(filename)
                    ds['class'].append(CLASSES.index(d))
                    ds['label'].append(d)
                    ds['duration'].append(duration)
                    ds['sr'].append(sr)
        data = pd.DataFrame(ds)
        data.to_csv(Path(self.filepaths,'metadata.csv'), index=False)
        logging.info(f"{data.shape[0]} files read successfully")
        return data
    def normalize_frames(self,m, epsilon=1e-12):
        return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]
    @staticmethod
    def applyWaveAugmentation(augment1,augment2):
        augmentations = []
        for i in range(len(augment1)):
            for j in range(len(augment2)):
                augmentations.append(Compose([augment1[i], augment2[j]]))
        return augmentations

    def pad_mfcc(self,mfcc, max_length):  # num_frames, nfilt=64.
        if len(mfcc) < max_length:
            mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
        return mfcc

    def sample_from_mfcc(self,mfcc, max_length):
        if mfcc.shape[0] >= max_length:
            r = random.choice(range(0, len(mfcc) - max_length + 1))
            s = mfcc[r:r + max_length]
        else:
            s = self.pad_mfcc(mfcc, max_length)
        return np.expand_dims(s, axis=-1)

    def read(self,filename, sample_rate=config.dataparam['SAMPLE_RATE']):
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        assert sr == sample_rate
        return audio

    def read_mfcc(self,audio, sample_rate):
        # audio = self.read(input_filename, sample_rate)
        energy = np.abs(audio)
        silence_threshold = np.percentile(energy, 95)
        offsets = np.where(energy > silence_threshold)[0]
        audio_voice_only = audio[offsets[0]:offsets[-1]]
        mfcc = self.mfcc_fbank(audio_voice_only, sample_rate)
        return mfcc

    def mfcc_fbank(self,signal: np.array, sample_rate: int):  # 1D signal array.
        # Returns MFCC with shape (num_frames, n_filters, 3).
        filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=config.dataparam['NUM_FBANKS'])
        frames_features = self.normalize_frames(filter_banks)
        return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.

    def trim_silence(self,audio, threshold):
        """Removes silence at the beginning and end of a sample."""
        energy = librosa.feature.rms(audio)
        frames = np.nonzero(np.array(energy > threshold))
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        audio_trim = audio[0:0]
        left_blank = audio[0:0]
        right_blank = audio[0:0]
        if indices.size:
            audio_trim = audio[indices[0]:indices[-1]]
            left_blank = audio[:indices[0]]  # slice before.
            right_blank = audio[indices[-1]:]  # slice after.
        return audio_trim, left_blank, right_blank

    def aug_func(self,x):
        aug_funcs = self.waveaugment
        auglist = []
        for func in random.sample(aug_funcs,2):
            auglist.append(func)
        augment = Compose(auglist)
        return augment(samples=x, sample_rate=config.dataparam['SAMPLE_RATE'])



    def build_dataset(self,augmentations, keepdims=True,augment = False,random_augmentation=False,n_augmented_samples = 1, crop_dims= (128,128),outpath=config.filepaths['file_path']):
        X = [] #stores the computed db scaled power spectrum of n second audio segments
        label = []
        Z = [] #stores waveforms of the n second segments
        CLASSES =  self.get_classes()
        for d in os.listdir(self.filepaths):
            temp =  Path(self.filepaths, d)
            if os.path.isdir(temp) and len(os.listdir(temp)) > 0:
                for file in temp.glob("**/*.wav"):
                    # filename =  Path(temp,file)
                    y,sr = librosa.load(file, sr=None)
                    if augment:
                        if random_augmentation:
                            if n_augmented_samples > 1:
                                ylist = [y] + [self.aug_func(y) for i in range(n_augmented_samples)]
                            else:
                                ylist = [self.aug_func(y)]
                        else:
                            if n_augmented_samples > 1:
                                ylist = [y] + [augmentations(samples = y, sample_rate = sr) for i in range(n_augmented_samples)]
                            else:
                                ylist = [augmentations(samples = y, sample_rate = sr)]
                    else:
                        ylist = [y]
                    for k in range(len(ylist)):
                        y =  ylist[k]
                        Z.append(y)
                        mfcc= self.read_mfcc(y,config.dataparam['SAMPLE_RATE'])
                        smfcc = self.sample_from_mfcc(mfcc, config.dataparam['NUM_FRAMES'])
                        if not keepdims:
                            X.append(cv2.resize(smfcc, crop_dims, interpolation = cv2.INTER_AREA))
                        else:
                            X.append(smfcc)
                        label.append(CLASSES.index(d))
        np.savez(outpath,x =  np.array(X), y = np.array(label), z= np.array(Z), c = CLASSES)
        logging.info(f"{len(X)} audio samples created successfully")
    
    def build_ds(self,augmentations, keepdims=True,augment = False,random_augmentation=False,n_augmented_samples = 1, crop_dims= (128,128),outpath=config.filepaths['file_path']):
        CLASSES =  self.get_classes()
        if Path(outpath).is_file():
            logging.info(f"{outpath} already exists")
            Path(outpath).unlink(missing_ok=True)
        hf = h5py.File(Path(outpath), 'w')
        counter = 0
        for d in os.listdir(self.filepaths):
            temp =  Path(self.filepaths, d)
            if os.path.isdir(temp) and len(os.listdir(temp)) > 0:
                for file in temp.glob("**/*.wav"):
                    # filename =  Path(temp,file)
                    y,sr = librosa.load(file, sr=None)
                    if augment:
                        if random_augmentation:
                            if n_augmented_samples > 1:
                                ylist = [y] + [self.aug_func(y) for i in range(n_augmented_samples)]
                            else:
                                ylist = [self.aug_func(y)]
                        else:
                            if n_augmented_samples > 1:
                                ylist = [y] + [augmentations(samples = y, sample_rate = sr) for i in range(n_augmented_samples)]
                            else:
                                ylist = [augmentations(samples = y, sample_rate = sr)]
                    else:
                        ylist = [y]
                    for k in range(len(ylist)):
                        y =  ylist[k]
                        mfcc= self.read_mfcc(y,config.dataparam['SAMPLE_RATE'])
                        smfcc = self.sample_from_mfcc(mfcc, config.dataparam['NUM_FRAMES'])
                        if not keepdims:
                            smfcc = cv2.resize(smfcc, crop_dims, interpolation = cv2.INTER_AREA)
                        data = np.expand_dims(smfcc, axis = 0)
                        ylabel = np.array([CLASSES.index(d)])
                        signal = np.expand_dims(y, axis = 0)
                        if counter == 0:
                            hf.create_dataset('X', data = data, maxshape=(None, data.shape[1], data.shape[2],1), chunks=True, compression="gzip")
                            hf.create_dataset('label', data = ylabel, maxshape=(None,), chunks=True, compression="gzip")
                            # hf.create_dataset('signal', data = signal, maxshape=(None,signal.shape[1] * 3), chunks=True, compression="gzip")
                            hf.create_dataset('CLASSES', data = CLASSES, maxshape=(None,), chunks=True, compression="gzip")
                        else:
                            hf['X'].resize((hf['X'].shape[0] + data.shape[0]), axis = 0)
                            hf['X'][-data.shape[0]:] = data

                            hf['label'].resize((hf['label'].shape[0] + ylabel.shape[0]), axis = 0)
                            hf['label'][-ylabel.shape[0]:] = ylabel

                            # hf['signal'].resize((hf['signal'].shape[0] + signal.shape[0]), axis = 0)
                            # hf['signal'][-signal.shape[0]:] = signal
                        counter += 1
        hf.close()
class Augmenter:
    def __init__(self,in_dir, out_dir, n_augmented_samples=1):
        self.in_dir =  in_dir
        self.out_dir =  out_dir
        self.n_augmented_samples = n_augmented_samples
        self.waveaugment = [AddGaussianNoise(min_amplitude=0.001,max_amplitude=0.015, p=0.5),TimeStretch(min_rate=0.8,max_rate=1.25,p=0.5),PitchShift(min_semitones=-4,\
            max_semitones=4,p=0.5),Shift(min_fraction=-0.5,max_fraction=0.5, p=0.5)]
    def aug_func(self,x,sr):
        aug_funcs = self.waveaugment
        auglist = []
        for func in random.sample(aug_funcs,2):
            auglist.append(func)
        augment = Compose(auglist)
        return augment(samples=x, sample_rate=sr)
    def build_dataset(self):
        input_dir =  self.in_dir
        out_dir =  self.out_dir
        n_augmented_samples = self.n_augmented_samples
        for dir in  os.listdir(input_dir):
            cur_dir = Path(input_dir,dir)
            cur_ds_dir = Path(out_dir,dir)
            if cur_ds_dir.is_dir():
                shutil.rmtree(cur_ds_dir)
            cur_ds_dir.mkdir(parents=True, exist_ok=True)
            print(f"Currently augmenting class={dir} containing {len(os.listdir(cur_dir))} samples")
            for f in cur_dir.iterdir():
                if f.is_file() and   f.suffix  == '.wav':
                    y,sr =  librosa.load(f, sr=None)
                    if n_augmented_samples > 1:
                        ylist = [y] + [self.aug_func(y,sr) for i in range(n_augmented_samples)]
                    else:
                        ylist = [self.aug_func(y,sr)]
                for k in range(len(ylist)):
                    y =  ylist[k]
                    new_name = f.name.split(".")[0] + "_" + str(k) + ".wav"
                    sf.write(Path(cur_ds_dir, new_name), y, sr)
                    # print(Path(cur_ds_dir, new_name))
        print("Augmentation completed successfully")


class Visualize:
    def __init__(self) -> None:
        self.fig_path = config.filepaths['fig_path']
    def plot_components(self,X_pca,y,CLASSES, x_str=r'$x_1$', y_str=r'$x_2$',save=False, filename="figure.png"):
        fig = plt.figure(1,figsize=(10,6))
        # plt.style.use("seaborn")
        ax =  fig.add_subplot(111)
        scatter = ax.scatter(X_pca[:,0],X_pca[:,1], c=list(y))
        ax.set_xlabel(x_str)
        ax.set_ylabel(y_str)
        # ax.set_zlabel("PCA Component 3")
        ax.legend(handles=scatter.legend_elements()[0], labels=CLASSES,bbox_to_anchor=(1.2, 1.0))
        # plt.grid()
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.fig_path}/{filename}', bbox_inches ="tight", dpi=300)
        plt.show()
    
    def plot_confusion_matrix(self,data,labels,save=False, filename="heatmap.png"):
        df_cm = pd.DataFrame(data, index = labels, columns = labels)
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True,cmap='Blues',annot_kws={"size": 14},fmt=".2f",cbar=True) #cmap="YlGnBu"
        if save:
            plt.savefig(f'{self.fig_path}/{filename}', bbox_inches ="tight", dpi=300)
        plt.show()
    def plot_scatter(self,X,y, CLASSES, s=5,n=10,fs=6,figsize=(8,6),save=False,fname='figure.png'):
        _, ax = plt.subplots(figsize=figsize)
        ax.scatter(X[:n,0], X[:n,1], c=list(y[:n]),s=s)
        ax.set_xlabel(r"$X_1$",fontsize=12)
        ax.set_ylabel(r"$X_2$",fontsize=12)
        z,w = X[:n,0],X[:n,1]
        for i, txt in enumerate(y[:n]):
            ax.annotate(CLASSES[txt].title(), (z[i], w[i]),fontsize=fs)
        if save:
            plt.savefig(f'{self.fig_path}/{fname}', bbox_inches ="tight", dpi=300)
        plt.show()

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

class Metrics:
    def __init__(self) -> None:
        pass
    @staticmethod
    def batch_cosine_similarity(x1, x2):
        # https://en.wikipedia.org/wiki/Cosine_similarity
        # 1 = equal direction ; -1 = opposite direction
        mul = np.multiply(x1, x2)
        s = np.sum(mul, axis=1)
        return s


class Wrangler:
  def __init__(self, base_dir, data_dir):
    self.data_dir =  data_dir
    self.base_dir =  base_dir
  def delete(self):
    if Path(self.base_dir, 'train').is_dir():
        shutil.rmtree(Path(self.base_dir, 'train'))
    if Path(self.base_dir, 'val').is_dir():
        shutil.rmtree(Path(self.base_dir, 'val'))
    if Path(self.base_dir, 'test').is_dir():
        shutil.rmtree(Path(self.base_dir, 'test'))
    logging.info("Data directories deleted successfully")
  def split_dataset(self, ratio = (.8, 0.1,0.1))->None:
    if Path(self.base_dir, 'train').is_dir()  or Path(self.base_dir, 'val').is_dir()  or Path(self.base_dir, 'test').is_dir():
        logging.info("Dataset already split into train and test/val/test directories") 
    else:
        logging.info(f"Splitting data into train/val/test into ratios {ratio}")
        splitfolders.ratio(self.data_dir, output=self.base_dir, seed=1337, ratio=ratio)
        logging.info(f"Data split completed successfully, output= {self.base_dir}")

class Builder:
    def __init__(self) -> None:
        # We are creating a set of audio data augmentation techniques from https://github.com/iver56/audiomentations 
        self.waveaugment = [AddGaussianNoise(min_amplitude=0.001,max_amplitude=0.015, p=0.5),TimeStretch(min_rate=0.8,max_rate=1.25, p=0.5),PitchShift(min_semitones=-4,max_semitones=4,p=0.5),\
            Shift(min_fraction=-0.5,max_fraction=0.5,p=0.5)]
        self.augmentations = Preprocessing.applyWaveAugmentation(self.waveaugment,self.waveaugment) 

    def build(self,input_dir, output_name,mode="pretrain", n_samples=2) -> None:
        pp = Preprocessing(datapath=input_dir)
        pp.build_ds(self.augmentations[0],keepdims=True,augment=True,random_augmentation=True,n_augmented_samples=n_samples,crop_dims= (128,128),outpath=Path(input_dir, output_name + ".h5"))


if __name__ == "__main__":
    logging.info("Program started successfully")

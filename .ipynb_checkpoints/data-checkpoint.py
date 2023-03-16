from __future__ import print_function
import random
import logging
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import soundfile as sf
import conf as config
import h5py 
import os
from pathlib import Path
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class DataLoader:
    def __init__(self, datapath = "") -> None:
        if(datapath == ""):
            self.datapath =  config.filepaths['file_path']
        else:
            self.datapath = datapath

    def load(self):
        if Path(self.datapath).exists():
            logging.info(f"Loading data from {self.datapath}")
            if str(self.datapath).endswith('.npz'):
                data =  np.load(self.datapath, allow_pickle=True)
                self.X = data['x']
                self.y = data['y']
                self.Z =  data['z']
                self.CLASSES = list(data['c'])
                self.input_shape = self.X.shape[1:]
            elif str(self.datapath).endswith('.h5'):
                hf =  h5py.File(self.datapath, mode="r")
                self.X = np.array(hf.get('X'))
                self.y = np.array(hf.get('label'))
                self.CLASSES = [x.decode("utf-8") for x in list(hf.get('CLASSES'))]
                self.input_shape = self.X.shape[1:]
            self.X_ =  (self.X - self.X.mean(axis=0, keepdims=True))/self.X.std(axis=0,keepdims=True)
            logging.info(f"Data loaded successfully n_samples = {self.X.shape[0]}, n_classes = { len(np.unique(self.y)) }")
        else:
            logging.error(f"File {self.datapath} does not exist")


class Chunk:
    def __init__(self):
        pass 
    def create_dataset(self,base_dir: str, data_name: str,  dur=2)->None:
      outpath =  Path(base_dir, f'{data_name}_{dur}')
      outpath.mkdir(parents=True, exist_ok=True)
      data_dir = Path(base_dir, data_name)
      print(data_dir)
      for d in os.listdir(data_dir):
          current_path = Path(outpath, d)
          current_path.mkdir(parents=True, exist_ok=True)
          temp = Path(data_dir,d)
          if(os.path.isdir(temp)):
              for file in temp.glob("**/*.wav"):
                  filename =  Path(temp,file)
                  y,sr = sf.read(filename)
                  stepsize = int(sr * dur)
                  # print(f"Currently splitting {filename} in {d} class")
                  for idx in range(0, len(y),  stepsize):
                    x = y[idx :  idx + stepsize ]
                    if(len(x) >=  stepsize):
                      newfilename = filename.name.split('.')[0] + str(idx) + ".wav"
                      sf.write(Path(current_path,newfilename), x, sr, subtype='PCM_16')
      
 
if __name__ == "__main__":
    logging.info("Program started successfully")
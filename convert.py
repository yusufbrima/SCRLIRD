"""
author: Yusuf Brima
This file contains a function for converting/saving audio\
     samples from an input dir to an output in a stanadardized format.
This format is as follows:
 input_dir
  -class_1
      -file_1
      -file_2
      - ...
      -file_n
   -class_2
      -file_1
      -file_2
      - ...
      -file_n
  - ...
  -class_n
      -file_1
      -file_2
      - ...
      -file_n
"""
import shutil
from pathlib import Path
import os
import random
import logging
import warnings
import argparse
import tqdm
import numpy as np
np.set_printoptions(precision=4)
warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

parser =  argparse.ArgumentParser()

parser.add_argument("--input_dir", type=str, required=True,  help="This is the path to the input audio dataset.\
     It can be relative if it is located in the current working directory or fully qualified path otherwise.")
parser.add_argument("--output_dir", type=str, required=True,  help="This is the path to the output audio dataset.\
     It can be relative if it is located in the current working directory or fully qualified path otherwise.")

args =  parser.parse_args()


def build_dataset(input_dir:str, out_dir:str) -> None:
    """
     This function takes two input arguments
     input_dir: str
     out_dir: str

     Create a directory system that resembles the class labels for the chosen dataset and copies the files if they have .wav extensions and covert
     them to .wav otherwise before writing them to the new output location by calling the ffmpeg process. To learn more about ffmpeg, please see the documentation 
     here: https://ffmpeg.org/
    
    """
    for dir in  tqdm.tqdm(os.listdir(input_dir)):
        cur_dir = Path(input_dir,dir)
        cur_ds_dir = Path(out_dir,dir)
        if cur_ds_dir.is_dir():
            shutil.rmtree(cur_ds_dir)
        cur_ds_dir.mkdir(parents=True, exist_ok=True)
        for f in cur_dir.iterdir():
            if f.is_file():
                if f.suffix  != '.wav':
                    new_fname = Path(cur_ds_dir, f.name.split(".")[0] + f'_{f.stem}.wav' )
                    logging.info(f"Coping {f.name} to {new_fname.name}")
                    os.system(f'ffmpeg -i {str(f)} {str(new_fname)}')

            else:
                i = 0
                for item in f.iterdir():
                    if item.is_file() and   item.suffix  != '.wav':
                        new_fname = Path(cur_ds_dir, item.name.split(".")[0] + f"_{f.stem}_{i}_"+ '.wav' )
                        logging.info(f"Coping {item.name} to {new_fname.name}")
                        os.system(f'ffmpeg -i {str(item)} {str(new_fname)}')
                    else:
                        new_fname = Path(cur_ds_dir, item.name.split(".")[0] + f"_{f.stem}_{i}_"+ '.wav' )
                        logging.info(f"Coping {item.name} to {new_fname.name}")
                        shutil.copy2(item, new_fname)
                    i +=1
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Program started successfully")
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    build_dataset(input_dir=input_dir, out_dir=out_dir)


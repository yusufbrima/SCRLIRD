
# Supervised Contrastive Representation Learning for Individual Recognition in the Wild

This codebase is a work-in-progress. Full details tba.

### Sample Results
<img src="https://github.com/yusufbrima/SCRLIRD/blob/master/Figures/1.gif" style="width:60%;" alt="Learnt 2D and 3D principal component latent representations of both human and non-human primates.">
<img src="https://github.com/yusufbrima/SCRLIRD/blob/master/Figures/3.gif" style="width:60%;" alt="Learnt 2D and 3D principal component latent representations of speeches at the United States Congress by five world leaders.">

 *Model name* | *Testing dataset* | *Num speakers* | *Top-1 Accuracy* | *Top-3 Accuracy* | Download model
 | :--- | :--- | :--- | :--- | :--- | :--- | 

# Run Locally

Clone the project

```bash
  git clone https://github.com/yusufbrima/SCRLIRD
```

Go to the project directory

```bash
  cd SCRLIRD
```

## Install dependencies

### Prerequisites

#### Requirements
- tensorflow>=2.0
- keras>=2.3.1
- python>=3.6

### PIP installation
```bash
  pip i install -r requirements.txt
```
### Conda installation
```bash
  conda env create -f environment.yml
```


### Building a pre-training dataset

```bash
  export CUDA_VISIBLE_DEVICES=0; python3 train.py build --input_dir /your/input/data/path --output preprocessing_output_path --n nunmer_0f_augmented_samples
```
### Pre-training and fine-tuning

```bash
  export CUDA_VISIBLE_DEVICES=0; python3 train.py train --p_input_dir /your/pretraining/input/data/path --p_output  pre_training_file_name  --d_input_dir /your/finetuning/input/data/path --d_output directory_name_for_splitting --pbs pre_training_batch_size --dbs fine_tuning_batch_size --epochs number_of_epochs --ft flag_to_finetune --npt nunmer_0f_pre_training_augmented_samples --nds nunmer_0f_fine_tuning_augmented_samples
```

* Run with pretrained model

```python
from __future__ import print_function
import random
import warnings
from pathlib import Path
import os
import librosa
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
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
    y,sr = librosa.load(file , sr = None)

    # mfcc =  pp.read_mfcc(y, sample_rate=sr)

    smfcc = pp.sample_from_mfcc(pp.read_mfcc(y, sample_rate=sr), config.dataparam['NUM_FRAMES']) 
    return np.expand_dims(smfcc, axis=0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Extracting features of speaker 1
    f1 = Path("Samples/Speaker_0_0.wav")
    f2 = Path("Samples/Speaker_0_1.wav")

    X1 = extract_features(f1)
    X2 = extract_features(f2)



    # Extracting features of speaker 2
    f3 = Path("Samples/Speaker_1_1.wav")
    f4 = Path("Samples/Speaker_1_0.wav")
    X3 = extract_features(f3)
    X4 = extract_features(f4)

    model_paths = "/home/staff/y/ybrima/Desktop/scratch/winter/valid_until_31_July_2023/ybrima/data/Models/LibriSpeech100/Prototype/Pretrain_Size/"

    model =  tf.keras.models.load_model(Path(model_paths,'ResNet50_InfoNCE_encoder'), compile=False)

    X = model(X1)
    X2 = model(X2)
    X3 = model(X3)
    X4 = model(X4)

    print(f"Same speaker {cosine_similarity(X, X2),cosine_similarity(X3, X4)}, different speakers{cosine_similarity(X, X3),cosine_similarity(X, X4)}")



```
## Authors

- [@yusufbrima](https://www.github.com/yusufbrima)


## Acknowledgements
 - [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)
 - [Audiomentations](https://github.com/iver56/audiomentations)
 - [DeepSkeaker](https://github.com/philipperemy/deep-speaker)
## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
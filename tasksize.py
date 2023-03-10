#!/usr/bin/env python3
# qsub -v PATH -v PYTHONPATH=$PWD -cwd -l mem=8G,cuda=1 Playground.py
from __future__ import print_function
import argparse
import random
import time 
import datetime
import warnings
from pathlib import Path
from cgi import test
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import pandas as pd
import logging
from sklearn.utils import shuffle
import conf as config 
from util import Preprocessing,SupervisedContrastiveLoss,Wrangler,Builder
from data import DataLoader
from models import SCRL
warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



class Trainer:
    def __init__(self, model, model_name = "ResNet50") -> None:
        """
        This method instantiate the Trainer class
        :param model: tf.keras.applications model to be trained
        :param model_name: string name for the instantiated model
        """
        self.m_name =model_name
        self.model = model

    def finetune(self, d_input_dir = config.filepaths['base_path'],d_output =  None, n_augmentations=1,\
        batch_size = config.learningparams['ft_batch_size'], epochs = config.learningparams['num_epochs'], save=False) -> None:
        """
        This function trains a baseline model and fine-tune the pre-trained models
        :param d_input_dir: The path to the input downstream dataset
        :param d_output: The path to the output train/val/test splits
        :param b_augmentations: The integer number of waveform augmentations to apply to the train samples
        :param batch_size: The fine-tuning batch size default is 32
        :param epochs: The number of model fine-tuning epoches
        :param save: Boolean flag to save the fine-tuned model
        """
        ### Here we are looping over the augmentations
        metrics = ['baseline_top_1_accuracy',  'baseline_top_3_accuracy','tripplet_loss_top_1_accuracy',\
             'tripplet_loss_top_3_accuracy', 'infoNCE_top_1_accuracy', 'infoNCE_top_3_accuracy' ]

        results = {'size':[], \
            f'{self.m_name}_baseline_top_1_accuracy': [],f'{self.m_name}_baseline_top_3_accuracy': [],\
            f'{self.m_name}_tripplet_loss_top_1_accuracy': [],f'{self.m_name}_tripplet_loss_top_3_accuracy': [],\
            f'{self.m_name}_infoNCE_top_1_accuracy': [],f'{self.m_name}_infoNCE_top_3_accuracy': [],
            'starttime': [],'endtime': []}
        if d_output == None:
            config.filepaths['datapath_path']
        else:
            if not Path(d_output).is_dir():
                d_output =  Path(d_input_dir,d_output)
                # d_output.mkdir(exist_ok=True)
        
        wr = Wrangler(base_dir=d_input_dir, data_dir=d_output)
        logging.info(d_output)
        factor =  np.array([10,20,30,40,50,60,70,80,90,100])
        
        i = 0
        pre_train_path = Path(args.p_input_dir, args.p_output + ".h5")
        if pre_train_path.is_file():
            dl2 =  DataLoader(datapath=pre_train_path)
            dl2.load()
        else:
            ds_bld = Builder()
            ds_bld.build(input_dir= args.p_input_dir,output_name= args.p_output, n_samples=args.npt)
            dl2 =  DataLoader(datapath=pre_train_path)
            dl2.load()
        logging.info(dl2.X.shape)
        #Contrastive Learning phase infoNCE
        infoNCE_scrl = SCRL(dl2.input_shape,len(dl2.CLASSES),model=models['ResNet50'])
        infoNCE_encoder = infoNCE_scrl.create_encoder()
        infoNCE_encoder_with_projection_head = infoNCE_scrl.add_projection_head(infoNCE_encoder)
        infoNCE_encoder_with_projection_head.compile(optimizer=tf.keras.optimizers.Adam(config.learningparams['learning_rate']),loss=SupervisedContrastiveLoss(config.learningparams['temperature']))
        infoNCE_encoder_with_projection_head.summary()
        history = infoNCE_encoder_with_projection_head.fit(x=dl2.X, y=dl2.y, batch_size=config.learningparams['batch_size'], epochs=epochs)


    
        #Contrastive Learning phase TrippletLoss
        TrippletLoss_scrl = SCRL(dl2.input_shape,len(dl2.CLASSES),model=models['ResNet50'])
        TrippletLoss_encoder = TrippletLoss_scrl.create_encoder()
        TrippletLoss_encoder_with_projection_head = TrippletLoss_scrl.add_projection_head(TrippletLoss_encoder)
        TrippletLoss_encoder_with_projection_head.compile(optimizer=tf.keras.optimizers.Adam(config.learningparams['learning_rate']),loss=tfa.losses.TripletHardLoss())
        TrippletLoss_encoder_with_projection_head.summary()
        history = TrippletLoss_encoder_with_projection_head.fit(x=dl2.X, y=dl2.y, batch_size=config.learningparams['batch_size'], epochs=epochs)

            

        for f in factor/100:
            ts = time.time()
            dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            results['starttime'].append(dt)
            ds_bld = Builder()
            wr.split_dataset(ratio= (f, 1-f))

            wr2 = Wrangler(base_dir=Path(d_input_dir,'final'), data_dir=Path(d_input_dir, "train"))
            wr2.split_dataset(ratio= (.8, 0.1,0.1))
            
        #     # We are preprocessing the train dataset 
            pp  = Preprocessing(datapath=Path(Path(d_input_dir,'final'), 'train'))
        #     # df =  pp.get_files()
    
            pp.build_ds(ds_bld.augmentations[0], keepdims=True,augment = True,random_augmentation=True,n_augmented_samples = 2, crop_dims= (128,128),outpath=Path(Path(d_input_dir,'final'), 'train/train.h5'))
            
            train_dl =  DataLoader(datapath= Path(Path(d_input_dir,'final'), 'train/train.h5'))

            train_dl.load()

            logging.info(f"Training shape {train_dl.X.shape}")


            
        #     # We are preprocessing the validation dataset 
            pp  = Preprocessing(datapath=Path(Path(d_input_dir,'final'), 'val'))
        #     # df =  pp.get_files()
            pp.build_ds(ds_bld.augmentations[0], keepdims=True,augment = False,random_augmentation=False,n_augmented_samples = 1, crop_dims= (128,128),outpath=Path(Path(d_input_dir,'final'), 'val/val.h5'))

            val_dl =  DataLoader(datapath= Path(Path(d_input_dir,'final'), 'val/val.h5'))

            val_dl.load()

            logging.info(f"Validation shape {val_dl.X.shape}")

            
        #     # We are preprocessing the test dataset 
            pp  = Preprocessing(datapath=Path(Path(d_input_dir,'final'), 'test'))
        #     # df =  pp.get_files()
            
            pp.build_ds(ds_bld.augmentations[0], keepdims=True,augment = False,random_augmentation=False,n_augmented_samples = 1, crop_dims= (128,128),outpath=Path(Path(d_input_dir,'final'), 'test/test.h5'))


            test_dl =  DataLoader(datapath= Path(Path(d_input_dir,'final'), 'test/test.h5'))

            test_dl.load()

            logging.info(f"Test set shape {test_dl.X.shape}")

            # We are training the baseline model
                        # We are creating a baseline w/o pre-training on the upstream dataset
            scrl = SCRL(train_dl.input_shape,len(train_dl.CLASSES), model=self.model)
            encoder = scrl.create_encoder()
            classifier = scrl.create_classifier(encoder)
            X, y = shuffle(train_dl.X, train_dl.y)
            classifier.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, validation_data=(val_dl.X, val_dl.y) )
            top_1_accuracy,top_3_accuracy = classifier.evaluate(test_dl.X, test_dl.y)[1:]
            print(f"Test accuracy before : {round(top_1_accuracy * 100, 2)}%, Top 3 accuracy {round(top_3_accuracy * 100, 2)}%")
            results[f'{self.m_name}_{metrics[0]}'].append(top_1_accuracy)
            results[f'{self.m_name}_{metrics[1]}'].append(top_3_accuracy)
            
            if save:
                classifier.save(f"{Path(config.filepaths['model_path'], f'Downstream_Size/{i}/{self.m_name}_Baseline_Classifier_Chimp_test')}")
                logging.info(f"Classifier saved to {Path(config.filepaths['model_path'], f'Downstream_Size/{i}/{self.m_name}_{type}_encoder_test')} successfully")


            #Contrastive Learning phase infoNCE
            classifier = infoNCE_scrl.create_classifier(infoNCE_encoder, trainable=False, num_classes= len(train_dl.CLASSES))
            history = classifier.fit(x=X, y=y, batch_size=config.learningparams['ft_batch_size'], epochs=epochs,validation_data=(val_dl.X, val_dl.y))
            top_1_accuracy,top_3_accuracy = classifier.evaluate(test_dl.X, test_dl.y)[1:]
            print(f"Test accuracy infoNCE after : {round(top_1_accuracy * 100, 2)}%, Top 3 accuracy {round(top_3_accuracy * 100, 2)}%")
            results[f'{self.m_name}_{metrics[4]}'].append(top_1_accuracy)
            results[f'{self.m_name}_{metrics[5]}'].append(top_3_accuracy)
            classifier.save(f"{Path(config.filepaths['model_path'], f'Downstream_Size/{i}/InfoNCE_Classifier')}")
            infoNCE_encoder_with_projection_head.save(f"{Path(config.filepaths['model_path'], f'Downstream_Size/{i}/{self.m_name}_InfoNCE_encoder')}")

            #Contrastive Learning phase trippletLossstartsize=config.learningparams['batch_size'], epochs=config.learningparams['num_epochs'], validation_split=0.2)
            classifier = TrippletLoss_scrl.create_classifier(TrippletLoss_encoder, trainable=False, num_classes= len(train_dl.CLASSES))
            history = classifier.fit(x=X, y=y, batch_size=config.learningparams['ft_batch_size'], epochs=epochs,validation_data=(val_dl.X, val_dl.y))
            top_1_accuracy,top_3_accuracy = classifier.evaluate(test_dl.X, test_dl.y)[1:]
            print(f"Test accuracy trippletLoss after : {round(top_1_accuracy * 100, 2)}%, Top 3 accuracy {round(top_3_accuracy * 100, 2)}%")
            results[f'{self.m_name}_{metrics[2]}'].append(top_1_accuracy)
            results[f'{self.m_name}_{metrics[3]}'].append(top_3_accuracy)
            classifier.save(f"{Path(config.filepaths['model_path'], f'Downstream_Size/{i}/Tripplet_loss_Classifier')}")
            TrippletLoss_encoder_with_projection_head.save(f"{Path(config.filepaths['model_path'], f'Downstream_Size/{i}/{self.m_name}_Tripplet_loss_encoder')}")

            results['size'].append(i)
            ts = time.time()
            dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            results['endtime'].append(dt)
            print(results)
            rdf = pd.DataFrame(results)
            logging.info(rdf)
            rdf.to_csv(Path(f"./Data/Downstream_Size_Result_{self.m_name}_{args.d_output}.csv"))
            logging.info(Path(f"./Data/Downstream_Size_Result_{self.m_name}_{args.d_output}.csv"))
            
            if(f != 1.0):
                wr.delete()
                wr2.delete()
            print(f, 1-f)
            i += 1

    def pretrain(self, loss_fn, dl_pt, save=False, bs = config.learningparams['batch_size'], epochs = config.learningparams['num_epochs'], m_type=["InfoNCE", "Triplet"]) -> None:
        """
        This function pre-trains the passed model to the constructor using the specified objective functions
        :param loss_fn: list of loss functions
        :param dl_pt: pre-training pre-processed compressed numpy array of X,y pairs
        :param save: a boolean flag to save the trained model
        :param bs: pre-training batch size, default is 265. This can be changed in the Conf.py or passed in the command line interface
        :param epochs: the integer number of epochs for pre-training
        :param type: list of pre-training objective names
        """
        self.pt_models = []
        self.pt_models_names = []
        self.pt_base_models = []
        for i in range(len(loss_fn)):
            scrl = SCRL(dl_pt.input_shape,len(dl_pt.CLASSES),model=self.model)
            encoder = scrl.create_encoder()
            self.encoder_with_projection_head = scrl.add_projection_head(encoder)
            self.encoder_with_projection_head.compile(optimizer=tf.keras.optimizers.Adam(config.learningparams['learning_rate']),loss= loss_fn[i])
            self.encoder_with_projection_head.summary()
            self.encoder_with_projection_head.fit(x=dl_pt.X, y=dl_pt.y, batch_size=bs, epochs=epochs)
            self.pt_models.append(encoder)
            self.pt_models_names.append(m_type[i])
            self.pt_base_models.append(scrl)
            if save:
                self.encoder_with_projection_head.save(f"{Path(config.filepaths['model_path'], f'{self.m_name}_{m_type[i]}_encoder_test')}")
                logging.info(f"Encoder saved to {Path(config.filepaths['model_path'], f'Downstream_Size/{self.m_name}_{m_type[i]}_encoder_test')} successfully")

# Build a parser
parser =  argparse.ArgumentParser()

# Build a subparser
subparser =  parser.add_subparsers(dest="command")

build =  subparser.add_parser("build")
build.add_argument("--input_dir", type=str, required=True, help="Pre-training data input directory path. It can be relative if it is located in the current working directory or fully qualified path otherwise.")
build.add_argument("--output", type=str, required=True, help="Pre-training data output file path. It can be relative if it is located in the current working directory or fully qualified path otherwise.")
build.add_argument("--n", type=int, required=False, default=2, help="Number of augmented samples to create.")


train = subparser.add_parser("train")
train.add_argument("--p_input_dir", type=str, required=True,  default=config.filepaths['pretrain_data_path'], help="Pre-training data input directory path. It can be relative if it is located in the current working directory or fully qualified path otherwise.")
train.add_argument("--p_output", type=str, required=True, default=config.filepaths['pretrain_file_path'] ,help="Pre-training data output file path. It can be relative if it is located in the current working directory or fully qualified path otherwise.")

train.add_argument("--d_input_dir", type=str, required=True, default=config.filepaths['base_path'], help="Downstream data input directory path. It can be relative if it is located in the current working directory or fully qualified path otherwise.")
train.add_argument("--d_output", type=str, required=True,default=config.filepaths['datapath_path'], help="Downstream data output file path. It can be relative if it is located in the current working directory or fully qualified path otherwise.")
train.add_argument("--pbs", default=265, type=int, required=False, help="Pre-training batch size, default=265")
train.add_argument("--npt", type=int, required=False, default=2, help="Number of augmented samples to create the pretraining dataset.")
train.add_argument("--nds", type=int, required=False, default=2, help="Number of augmented samples to create the downstream dataset.")
train.add_argument("--dbs", default=64, type=int, required=False, help="Fine-tuning batch size, default=64")
train.add_argument("--epochs", default=100, type=int, required=False, help="Number of epochs, default=100")
train.add_argument("--ft", default=0, type=int, required=False, help="Indicate whether in pre-training or in fine-tuning mode")

args =  parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Program started successfully")
    if args.command  == 'build':
        print("We are in the build subcommand")
        print(args.input_dir)
        #Here we want to pre-train a model using human speech dataset
        ds_bld =  Builder()
        ds_bld.build(input_dir= args.input_dir,output_name= args.output, n_samples=args.n)
    elif args.command == "train":
        models = {'ResNet50': tf.keras.applications.ResNet50} #{'VGG16': tf.keras.applications.VGG16}
        m_name = list(models.keys())[0]
        model_instance =  models[m_name]
        # We instantiate the Trainer class passing the current model and its name
        trainer =  Trainer(model=model_instance, model_name= m_name)
        infoNCE_loss_fn = SupervisedContrastiveLoss(config.learningparams['temperature'])
        triplet_loss_fn =  tfa.losses.TripletHardLoss()
        loss_fns = [infoNCE_loss_fn, triplet_loss_fn]
        loss_fn_names = ["InfoNCE", "Triplet"]
        pre_train_path = Path(args.p_input_dir, args.p_output + ".npz")
        if args.ft == 1:
            logging.info("Here we are fine-tuning the pre-trained model")
            trainer.finetune(d_input_dir=args.d_input_dir, d_output= args.d_output,n_augmentations=args.nds, batch_size=args.dbs, epochs= args.epochs)
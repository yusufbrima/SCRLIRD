from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import conf as config 
import random
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class SCRL:
    """
      SCRL --> Supervised Contrastive Representation Learning (SCRL)
    """
    def __init__(self,input_shape,num_classes, model=tf.keras.applications.ResNet50V2) -> None:
        self.input_shape = input_shape
        self.num_classes =  num_classes
        self.base_model = model 

    def create_encoder(self, custom = False):
        if custom:
           resnet = self.base_model
        else:
            resnet = self.base_model(
                include_top=False, weights=None, input_shape=self.input_shape, pooling="avg"
            )
            

        inputs = tf.keras.Input(shape=self.input_shape)
        # augmented = data_augmentation(inputs)
        outputs = resnet(inputs)
      
        outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="deepvocal-encoder")
        return model
    
    def create_classifier(self,encoder, trainable=True, num_classes =  0):
        for layer in encoder.layers:
            layer.trainable = trainable

        inputs = tf.keras.Input(shape=self.input_shape)
        features = encoder(inputs)
        features = tf.keras.layers.Dropout(config.learningparams['dropout_rate'])(features)
        features = tf.keras.layers.Dense(config.learningparams['hidden_units'], activation="relu")(features)
        features = tf.keras.layers.Dropout(config.learningparams['dropout_rate'])(features)
        if(num_classes == 0 ):
            outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(features)
        else:
            outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="deepvocal-classifier")

        model.compile(
            optimizer= tf.keras.optimizers.Adam(config.learningparams['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)],
        )
        return model
    
    def add_projection_head(self,encoder):
        inputs = tf.keras.Input(shape=self.input_shape)
        features = encoder(inputs)
        outputs = tf.keras.layers.Dense(config.learningparams['projection_units'], activation="relu")(features)
        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="deepvocal-encoder_with_projection-head"
        )
        return model
    
    def add_basetask(self,encoder, num_classes =  0):
        inputs = tf.keras.Input(shape=self.input_shape)
        features = encoder(inputs)
        features = tf.keras.layers.Dropout(config.learningparams['dropout_rate'])(features)
        outputs = tf.keras.layers.Dense(config.learningparams['projection_units'], activation="relu")(features)
        outputs =  tf.keras.layers.Dropout(config.learningparams['dropout_rate'])(features)
        if(num_classes == 0 ):
            outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(features)
        else:
            outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="deepvocal-basetask")

        # model = tf.keras.Model(
        #     inputs=inputs, outputs=outputs, name="deepvocal-encoder_with_projection-head"
        # )
        return model
if __name__ == "__main__":
    pass

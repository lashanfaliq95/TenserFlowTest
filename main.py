import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

np.set_printoptions(precision=3, suppress=True)


def load_dataset(name):
    if name == 'training':
        training_set = pd.read_csv("./training_dataset.csv")
        training_set = training_set.replace('?', np.nan)
        training_features = training_set.copy()
        training_labels = training_features.pop('Class')
        for col in training_features.columns.values:
            training_features[col] = pd.to_numeric(training_features[col])
            print(int(training_features[col].mean()))
            training_features[col] = training_features[col].fillna(int(training_features[col].mean()))

        training_features.to_excel('test.xlsx' , 'lashan')
        training_features = np.array(training_features)


        training_model = tf.keras.Sequential([
            layers.Dense(64),
            layers.Dense(1)
        ])

        training_model.compile(loss=tf.losses.MeanSquaredError(),
                               optimizer=tf.optimizers.Adam())
        training_model.fit(training_features, training_labels, epochs=10)

        training_model(training_features)
        print(training_labels, training_model(training_features))


if __name__ == '__main__':
    load_dataset('training')

import gc
import note_seq
import numpy as np
import tensorflow as tf

from pandas import read_csv
from argparse import ArgumentTypeError
from ddsp.training.data_preparation import prepare_tfrecord_lib as ddsp_lib

from ddsp_piano.data_processing.midi_encoders import MIDIRoll2Conditioning


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def dataset_from_csv(csv_path, split=None, **kwargs):
    """ Load dataset from a csv file.
    Returns:
        - dataset (tf.data.Dataset): tensorflow dataset from .csv
        - n_samples (int): number of dataset entries.
        - piano_models (list): list of different piano models in the dataset.
    """
    # .csv reading in pandas dataframe
    df = read_csv(csv_path, **kwargs)

    # Filter by split
    if split:
        df = df[df.split == split]

    # Convert dataframe to tf.data.Dataset
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            {key: df[key].values for key in df})
    )
    # Get dataset length and piano models
    n_samples = len(df)
    piano_models = df['year'].unique()

    return dataset, n_samples, piano_models


def collect_garbage():
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

import tensorflow as tf
from normalize_dataset import normalize

import argparse
from runpy import run_path



parser = argparse.ArgumentParser(description="Experiment Perameters")
parser.add_argument('training_samples', metavar='N', type=int)
parser.add_argument('model_path', metavar='N', type=str)
parser.add_argument('dataset_path', metavar='N', type=str)

args = parser.parse_args()

training_samples = args.training_rounds
model_path = args.model_path
dataset_path = args.dataset_path

model = tf.keras.models.load_model(model_path)


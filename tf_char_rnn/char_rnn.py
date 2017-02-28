""" A clean, no_frills character-level generative language model.
Created by Danijar Hafner, edited by Chip Huyen
for the class CS 20SI: "TensorFlow for Deep Learning Research"

Based on Andrej Karpathy's blog: 
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
from __future__ import print_function
import os
import time

import tensorflow as tf
import numpy as np
from sequence_predictor_model import SequencePredictor
from model_trainer import ModelTrainer

class Config:
    data_path = '../data/input.txt'
    save_path = 'checkpoints/blah4/char-rnn'
    summary_path = '.graphs/0entropy/test/makrov_20'
    init_from = None
    hidden_size = 32
    batch_size = 64
    max_length = batch_size
    lr = 0.0003
    save_every = 1000
    print_every = 1
    num_layers = 2
    num_epochs = 10
    vocab = ("ab")
    num_classes = len(vocab)


def main():
    
    config = Config();
    print(config.num_classes)
    GRUModel = SequencePredictor(config);
    Trainer = ModelTrainer(config, GRUModel)
    Trainer.do_training();
    
if __name__ == '__main__':
    main()

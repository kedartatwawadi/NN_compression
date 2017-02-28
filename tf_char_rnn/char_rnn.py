""" Based partially on CS224n assignment 3
"""
from __future__ import print_function
import os
import time
import argparse
import tensorflow as tf
import numpy as np
from sequence_predictor_model import SequencePredictor
from model_trainer import ModelTrainer

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/sequence_data/input.txt',
                        help='data directory containing input.txt')
    parser.add_argument('--info_path', type=str, default=None,
                       help='Information about the input file')
    parser.add_argument('--summary_path', type=str, default='.summary',
                       help='directory to store tf summary')
    parser.add_argument('--hidden_size', type=int, default=32,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for the training')
    parser.add_argument('--vocab', type=str, default="ab")    
    parser.add_argument('--print_every', type=int, default=1)                    
    
    return parser

def get_config_args():
    parser = get_argument_parser()
    config = parser.parse_args()
    config.max_length = config.batch_size
    config.num_classes = len(config.vocab)
    return config

def main():
    config = get_config_args()
    GRUModel = SequencePredictor(config);
    Trainer = ModelTrainer(config, GRUModel)
    Trainer.do_training();
    
if __name__ == '__main__':
    main()

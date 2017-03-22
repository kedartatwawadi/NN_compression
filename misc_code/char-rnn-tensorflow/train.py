from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
import json

from utils import TextLoader
from model import Model

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/sequence_data/input.txt',
                       help='data directory containing input.txt')
    parser.add_argument('--info_path', type=str, default=None,
                       help='Information about the input file')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--summary_dir', type=str, default='.summary',
                       help='directory to store tf summary')
    parser.add_argument('--output_path', type=str, default='outputs/output.txt',
                       help='Path to store the output loss and other details')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')                       
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    
    return parser





###############################################################################
# Train function
###############################################################################

def train(args):

    data_loader = TextLoader(args.data_path, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size
    args.file_size = data_loader.file_size
    print("Vocab size: ",args.vocab_size)
    print("File size: ",args.file_size)
    args.lower_bound = 0 #If we know the entropy then we set it to this
    data_info = {}
    if args.info_path is not None:
        assert os.path.isfile(args.info_path),"Info file not found in the path: %s"%args.info_path

        #Open the info file
        with open(args.info_path, 'rb') as f:
            data_info = json.load(f)
            #Assuming we know entropy
            args.lower_bound = data_info['Entropy']
            print(data_info)

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist 
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme
        
        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"
        
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)
        
    
    ##################################################
    # Get the model
    ##################################################
    model = Model(args)
    print("model Loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        writer = tf.summary.FileWriter(args.summary_dir,sess.graph)
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        ######################################################
        # Perform the training
        #####################################################
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer() #Need to check what this does
            state = sess.run(model.initial_state) #What is this initial state
            cumul_loss = 0
             
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                summary, train_loss, state, _ = sess.run([model.merged_summaries, model.cost, model.final_state, model.train_op], feed) #what is the training loss
                train_loss /= np.log(2)
                cumul_loss += train_loss
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

                if b%10 == 0:
                    writer.add_summary(summary,e*data_loader.num_batches + b)
             
            cumul_loss /= data_loader.num_batches
            print("Epoch {}: Cumulative Loss for the epoch: {:.3f}".format(e,cumul_loss))
            if (abs(cumul_loss - args.lower_bound) < 0.1):
                print("Stopping Training as we get a good loss.. :) ... ") 
                break    

        ##############################################################
        # Append details to the output file
        ##############################################################
        args.epoch_stopped=e+1
        args.last_epoch_loss = cumul_loss
        with open(args.output_path, 'a') as f:

            params = vars(args)
            params.update(data_info)
            #json.dump(params, f,indent=2)
            cPickle.dump(params,f)
            #f.write("\n ############################################# \n")

        with open(args.output_path+".json", 'a') as f:

            params = vars(args)
            params.update(data_info)
            json.dump(params, f,indent=2)
            #cPickle.dump(params)
            f.write("\n ############################################# \n")


def main():
    #############################################################
    # Parser for arguments
    parser = get_argument_parser()
    args = parser.parse_args()



    train(args)


if __name__ == '__main__':
    main()

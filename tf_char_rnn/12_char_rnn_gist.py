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
# # def get_argument_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default='data/sequence_data/input.txt',
#                        help='data directory containing input.txt')
#     parser.add_argument('--info_path', type=str, default=None,
#                        help='Information about the input file')
#     parser.add_argument('--save_dir', type=str, default='save',
#                        help='directory to store checkpointed models')
#     parser.add_argument('--summary_dir', type=str, default='.summary',
#                        help='directory to store tf summary')
#     parser.add_argument('--rnn_size', type=int, default=128,
#                        help='size of RNN hidden state')
#     parser.add_argument('--num_layers', type=int, default=2,
#                        help='number of layers in the RNN')
#     parser.add_argument('--model', type=str, default='gru',
#                        help='rnn or gru')
#     parser.add_argument('--batch_size', type=int, default=50,
#                        help='minibatch size')
#     parser.add_argument('--seq_length', type=int, default=50,
#                        help='RNN sequence length')
#     parser.add_argument('--num_epochs', type=int, default=50,
#                        help='number of epochs')
#     parser.add_argument('--save_every', type=int, default=1000,
#                        help='save frequency')
#     parser.add_argument('--learning_rate', type=float, default=0.0002,
#                        help='learning rate')
#     parser.add_argument('--decay_rate', type=float, default=0.97,
#                        help='decay rate for rmsprop')                       
#     parser.add_argument('--init_from', type=str, default=None,
#                        help="""continue training from saved model at this path. Path must contain files saved by previous training process """)
    
#     return parser



#DATA_PATH = '../data/chr1.fa'
DATA_PATH = '../data/input.txt'
SAVE_PATH = 'checkpoints/blah/char-rnn'
SUMMARY_PATH = 'graphs/0entropy'
INIT_FROM = None
HIDDEN_SIZE = 64
BATCH_SIZE = 64
NUM_STEPS = BATCH_SIZE
TEMPRATURE = 0.7
LR = 0.003
LEN_GENERATED = 100
SAVE_EVERY=1000
SAMPLE_EVERY=10
PRINT_EVERY = 1
NUM_EPOCHS=10
#VOCAB = (
#            " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#            "\\^_abcdefghijklmnopqrstuvwxyz{|}")
VOCAB = ("ab")

#################################################################################
# Util Functions
#################################################################################
def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window=NUM_STEPS, overlap=1, num_epochs=NUM_EPOCHS):
    for text in open(filename):
        text = vocab_encode(text, vocab)
        for _epoch in range(num_epochs):
            for start in range(0, len(text) - window, overlap):
                chunk = text[start: start + window]
                chunk += [0] * (window - len(chunk))
                yield chunk

def read_batch(stream, batch_size=BATCH_SIZE):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    #yield batch


######################################################################################
# Create the model
######################################################################################
def create_rnn(seq, hidden_size=HIDDEN_SIZE, num_layers=2):
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=False)
    in_state = tf.placeholder_with_default(
            cell.zero_state(tf.shape(seq)[0], tf.float32), [None, cell.state_size])
    # this line is to allow for dynamic batch size
    length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
    output, out_state = tf.nn.dynamic_rnn(cell, seq, length, in_state)
    return output, in_state, out_state


def create_model(seq, temp, vocab, hidden=HIDDEN_SIZE):
    seq = tf.one_hot(seq, len(vocab))
    output, in_state, out_state = create_rnn(seq, hidden)
    # fully_connected is syntactic sugar for tf.matmul(w, output) + b
    # it will create w and b for us
    logits = tf.contrib.layers.fully_connected(output, len(vocab), None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=seq[:, 1:]))
    # sample the next word from Maxwell-Boltzmann Distribution with temperature temp
    sample = tf.multinomial(tf.exp(logits[:, -1] / temp), 1)[:, 0]
    return loss, sample, in_state, out_state

def training(vocab, seq, loss, optimizer, global_step, temp, sample, in_state, out_state):
    saver = tf.train.Saver()
    start = time.time()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)
        sess.run(tf.global_variables_initializer())
        
        if INIT_FROM is not None:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(INIT_FROM))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
       
        state = None
        iteration = global_step.eval()
        for batch in read_batch(read_data(DATA_PATH, vocab)):
            
            feed = {seq: batch}
            if state is not None:
                feed.update({in_state: state})
            batch_loss, state, _ = sess.run([loss, out_state, optimizer], feed)
            batch_loss = batch_loss/np.log(2)
            if (iteration + 1) % PRINT_EVERY == 0:
                print('Iter {}:      Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))

            if (iteration + 1) % SAMPLE_EVERY == 0:
                online_intference(sess, vocab, seq, sample, temp, in_state, out_state)
            start = time.time()
            
            if (iteration + 1) % SAVE_EVERY == 0:
                saver.save(sess, SAVE_PATH, iteration)
            iteration += 1

def online_intference(sess, vocab, seq, sample, temp, in_state, out_state, seed='T'):
    seed = vocab[0]
    sentence = seed
    state = None
    for _ in range(LEN_GENERATED):
        batch = [vocab_encode(sentence[-1], vocab)]
        feed = {seq: batch, temp: TEMPRATURE}
        if state is not None:
            feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += vocab_decode(index, vocab)
    print(sentence)

def main():
    
    seq = tf.placeholder(tf.int32, [None, None])
    temp = tf.placeholder(tf.float32)
    loss, sample, in_state, out_state = create_model(seq, temp, VOCAB)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss, global_step=global_step)
    training(VOCAB, seq, loss, optimizer, global_step, temp, sample, in_state, out_state)
    
if __name__ == '__main__':
    main()

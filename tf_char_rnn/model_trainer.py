from __future__ import print_function
import os
import time

import tensorflow as tf
import numpy as np


class ModelTrainer():
    def vocab_encode(self, text):
        return [self.config.vocab.index(x) for x in text if x in self.config.vocab]

    def vocab_decode(self, array):
        return ''.join([self.config.vocab[x] for x in array])

    def read_data(self):
        window = self.config.max_length
        overlap = 1
        filename = self.config.data_path
        for text in open(filename):
            text = self.vocab_encode(text)
            for start in range(0, len(text) - window - 1, overlap):
                chunk = text[start: start + window + 1]
                chunk += [0] * (window+1 - len(chunk))
                yield chunk

    def get_batch(self,stream):
        input_batch = []
        label_batch = []
        for element in stream:
            input_batch.append(element[:-1])
            label_batch.append(element[1:]) 
            if len(label_batch) == self.config.batch_size:
                data_tuple = (input_batch, label_batch)
                yield data_tuple
                input_batch = []
                label_batch = []
        #yield batch

    def run_epoch(self,sess, epoch, writer=None):
        state = None
        for batch in self.get_batch(self.read_data()):
            _input = batch[0]
            _labels = batch[1]
            batch_loss, state, global_step, summary = self.model.train_on_batch(sess, _input , _labels, state)  

            writer.add_summary(summary, global_step)
            if (global_step + 1) % self.config.print_every == 0:
                print('Epoch: {} Global Iter {}:      Loss {}'.format(epoch, global_step, batch_loss) )
    
    def do_training(self):
        saver = tf.train.Saver()
        merged_summaries = tf.summary.merge_all()
    
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(self.config.summary_path, sess.graph)
            sess.run(tf.global_variables_initializer())
           
            for epoch in range(self.config.num_epochs):
                self.run_epoch(sess, epoch, writer);

            writer.close()


    def __init__(self, config, Model):
        self.config = config
        self.model = Model
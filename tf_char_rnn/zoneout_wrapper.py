from __future__ import print_function
import os
import time

import tensorflow as tf
import numpy as np
import sys



# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
  """Operator adding zoneout to all states (states+cells) of the given cell."""

  def __init__(self, cell, state_zoneout_prob, is_training=True, seed=None):
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not an RNNCell.")
    if (isinstance(zoneout_prob, float) and
        not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0)):
      raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                       % zoneout_prob)
    self._cell = cell
    self._zoneout_prob = zoneout_prob
    self._seed = seed
    self.is_training = is_training
  
  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
      raise TypeError("Subdivided states need subdivided zoneouts.")
    if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
      raise ValueError("State and zoneout need equally many parts.")
    output, new_state = self._cell(inputs, state, scope)
    if isinstance(self.state_size, tuple):
      if self.is_training:
          new_state = tuple((1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                        new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
                            for new_state_part, state_part, state_part_zoneout_prob in zip(new_state, state, self._zoneout_prob))
      else:
          new_state = tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                            for new_state_part, state_part, state_part_zoneout_prob in zip(new_state, state, self._zoneout_prob))
    else:
      if self.is_training:
          new_state = (1 - state_part_zoneout_prob) * tf.python.nn_ops.dropout(
                        new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
      else:
          new_state = state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
    return output, new_state


# Wrap your cells like this
#cell = ZoneoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_units, initializer=random_uniform(), state_is_tuple=True),
#zoneout_prob=(z_prob_cells, z_prob_states))
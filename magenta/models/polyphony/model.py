# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create a dataset from NoteSequence protos. """

import sequence
import logging
import sys
import tensorflow as tf
import numpy as np
from magenta.lib import note_sequence_io
import one_hot_delta_codec


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')

NUM_VOICES=6

class PolyphonicRNN(tf.contrib.learn.estimators.Estimator):
  pass

def gen_model():
  def model_fn(features, targets, mode, params):
    # there's only 1 data set, so we know it's all the same size
    sequence_lengths = tf.constant([features.get_shape()[1].value])

    # If state_is_tuple is True, the output RNN cell state will be a tuple
    # instead of a tensor. During training and evaluation this improves
    # performance. However, during generation, the RNN cell state is fed
    # back into the graph with a feed dict. Feed dicts require passed in
    # values to be tensors and not tuples, so state_is_tuple is set to False.
    state_is_tuple = True
    if mode == tf.contrib.learn.ModeKeys.INFER:
      state_is_tuple = False

    cells = []
    for num_units in params['layer_sizes']:
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
          num_units, state_is_tuple=state_is_tuple)

      if mode == tf.contrib.learn.ModeKeys.TRAIN:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=params['dropout_keep_prob'])

      cells.append(lstm_cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)

    initial_state = cell.zero_state(1, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell, features, sequence_lengths, initial_state, dtype=tf.float32,
        parallel_iterations=1, swap_memory=True)

    outputs_flat = tf.reshape(outputs, [-1, params['layer_sizes'][-1]])
    logits_flat = tf.contrib.layers.linear(outputs_flat, 130*NUM_VOICES)

    loss = None
    if mode != tf.contrib.learn.ModeKeys.INFER:
      labels_flat = tf.squeeze(targets, [0])

      total_cost = tf.constant([0], dtype=tf.float32)
      for voice in range(NUM_VOICES):
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.slice(logits_flat, [0,130*voice], [-1, 130]),
            tf.squeeze(tf.slice(labels_flat, [0,voice], [-1, 1]), [1]))
        total_cost = tf.add(total_cost, cost)
      loss = tf.reduce_mean(total_cost)

    predictions = []
    for voice in range(NUM_VOICES):
      prediction = tf.nn.top_k(tf.slice(logits_flat, [0,130*voice], [-1, 130]))
      predictions.append(prediction)

    train_op = None
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(
          loss=loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=params['learning_rate'],
          optimizer="Adam")

    return predictions, loss, train_op

  return model_fn

def main(unused_argv):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  reader = note_sequence_io.note_sequence_record_iterator(FLAGS.input)
  #np.set_printoptions(threshold=np.nan)
  note_sequence = reader.next()
  polyphonic_sequence = sequence.PolyphonicSequence(note_sequence)
  inputs, labels = one_hot_delta_codec.encode(
      polyphonic_sequence, max_voices=10, max_note_delta=30,
      max_intervoice_interval=50)

  estimator = tf.contrib.learn.Estimator(
      model_fn=gen_model(),
      model_dir='/tmp/polyphony',
      config=None,
      params={
          'layer_sizes': [128, 128],
          'dropout_keep_prob': .9,
          'learning_rate': .1,
      })

  estimator.fit(
      x=np.array([inputs]),
      y=np.array([labels]),
      steps=100)



if __name__ == "__main__":
  tf.app.run()

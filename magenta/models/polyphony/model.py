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
import sys
import tensorflow as tf
import numpy as np
from magenta.lib import note_sequence_io
import one_hot_delta_codec


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', '/tmp/polyphony_rnn',
                           'Directory to save model parameters, graph and etc.')
tf.app.flags.DEFINE_string('sequence_example_file', '',
                           'Path to TFRecord file containing '
                           'tf.SequenceExample records for training or '
                           'evaluation.')

class PolyphonicRNN(tf.contrib.learn.estimators.Estimator):
  pass

def gen_model(num_classes, classes_per_label):
  def model_fn(features, targets, mode, params):
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
        cell, features['inputs'], features['lengths'], initial_state,
        dtype=tf.float32, parallel_iterations=1, swap_memory=True)

    outputs_flat = tf.reshape(outputs, [-1, params['layer_sizes'][-1]])
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    loss = None
    if mode != tf.contrib.learn.ModeKeys.INFER:
      labels_flat = tf.squeeze(targets['labels'], [0])

      total_cost = tf.constant([0], dtype=tf.float32)
      for i in range(len(classes_per_label)):
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.slice(
                logits_flat,
                [0, sum(classes_per_label[0:i])],
                [-1, classes_per_label[i]]),
            tf.squeeze(tf.slice(labels_flat, [0, i], [-1, 1]), [1]))
        total_cost = tf.add(total_cost, cost)
      loss = tf.reduce_mean(total_cost)

    predictions = []
    for i in range(len(classes_per_label)):
      prediction = tf.nn.top_k(
          tf.slice(
              logits_flat,
              [0, sum(classes_per_label[0:i])],
              [-1, classes_per_label[i]]))
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

def _get_input_fn(mode, filename, input_size, labels_per_example, batch_size,
                  num_enqueuing_threads=4):
  def input_fn():
    file_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    sequence_features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                             dtype=tf.float32),
        'labels': tf.FixedLenSequenceFeature(shape=[labels_per_example],
                                             dtype=tf.int64)
    }

    _, sequence = tf.parse_single_sequence_example(
        serialized_example, sequence_features=sequence_features)

    # TODO: make this a queue?

    length = tf.shape(sequence['inputs'])[0]
    return (
        {
            'inputs': tf.expand_dims(sequence['inputs'], [0]),
            'lengths': tf.expand_dims(length, [0])
        }, {'labels': tf.expand_dims(sequence['labels'], [0])})

  return input_fn

def main(unused_argv):
  codec = one_hot_delta_codec.PolyphonyCodec()
  estimator = tf.contrib.learn.Estimator(
      model_fn=gen_model(codec.num_classes, codec.classes_per_label),
      model_dir=FLAGS.model_dir,
      config=tf.contrib.learn.RunConfig(save_summary_steps=10),
      params={
          'layer_sizes': [32, 32],
          'dropout_keep_prob': .9,
          'learning_rate': .001,
      })

  estimator.fit(
      input_fn=_get_input_fn(
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          filename=FLAGS.sequence_example_file,
          input_size=codec.input_size,
          labels_per_example=codec.labels_per_example,
          batch_size=2))



if __name__ == "__main__":
  tf.app.run()

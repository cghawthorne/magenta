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
from magenta.protobuf import music_pb2
from magenta.lib import midi_io


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', '/tmp/polyphony_rnn',
                           'Directory to save model parameters, graph and etc.')
tf.app.flags.DEFINE_string('sequence_example_train_file', '',
                           'Path to TFRecord file containing '
                           'tf.SequenceExample records for training.')
tf.app.flags.DEFINE_string('sequence_example_eval_file', '',
                           'Path to TFRecord file containing '
                           'tf.SequenceExample records for evaluation.')
tf.app.flags.DEFINE_bool('predict', False, 'If true, runs in predict mode.')

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

    batch_size = features['inputs'].get_shape()[0].value
    initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell, features['inputs'], features['lengths'], initial_state,
        dtype=tf.float32, parallel_iterations=1, swap_memory=True)

    # Separate this out into one layer per voice
    logits = []
    for voice in range(len(classes_per_label)):
      logits.append(tf.contrib.layers.linear(
          outputs,
          classes_per_label[voice],
          weights_regularizer=tf.contrib.layers.l2_regularizer(
              params['l2_regularizer_scale'])))

    loss = None
    if mode != tf.contrib.learn.ModeKeys.INFER:
      costs = []
      for voice in range(len(classes_per_label)):
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits[voice],
            tf.squeeze(
                tf.slice(
                    targets['labels'],
                    [0, 0, voice],
                    [-1, -1, 1]),
                [2]))
        costs.append(cost)
      total_cost = tf.add_n(costs, 'total_cost')
      loss = tf.reduce_mean(total_cost)

    prediction_indices = []
    for voice in range(len(classes_per_label)):
      _, prediction_index = tf.nn.top_k(logits[voice])
      prediction_indices.append(prediction_index)
    predictions = {
        'labels': tf.concat(2, prediction_indices),
    }

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

    length = tf.shape(sequence['inputs'])[0]

    queue = tf.PaddingFIFOQueue(
      capacity=1000,
      dtypes=[tf.float32, tf.int64, tf.int32],
      shapes=[(None, input_size), (None, labels_per_example), ()])

    enqueue_ops = [queue.enqueue([sequence['inputs'],
                                sequence['labels'],
                                length])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    batch = queue.dequeue_many(batch_size)

    return (
        {
            'inputs': batch[0],
            'lengths': batch[2],
        },
        {'labels': batch[1]}
    )

  return input_fn

def main(unused_argv):
  codec = one_hot_delta_codec.PolyphonyCodec()
  params = {
      'layer_sizes': [2048, 2048, 2048],
      'dropout_keep_prob': .9,
      'learning_rate': .00001,
      'batch_size': 5,
      'l2_regularizer_scale': .01,
  }

  # If predicting, don't do any dropout.
  if FLAGS.predict:
    params['dropout_keep_prob'] = 1.0

  estimator = tf.contrib.learn.Estimator(
      model_fn=gen_model(codec.num_classes, codec.classes_per_label),
      model_dir=FLAGS.model_dir,
      config=tf.contrib.learn.RunConfig(save_summary_steps=10),
      params=params)

  if not FLAGS.predict:
    train_input_fn=_get_input_fn(
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        filename=FLAGS.sequence_example_train_file,
        input_size=codec.input_size,
        labels_per_example=codec.labels_per_example,
        batch_size=params['batch_size'])

    eval_input_fn=_get_input_fn(
        mode=tf.contrib.learn.ModeKeys.EVAL,
        filename=FLAGS.sequence_example_eval_file,
        input_size=codec.input_size,
        labels_per_example=codec.labels_per_example,
        batch_size=params['batch_size'])

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=None,
        eval_steps=1,
        local_eval_frequency=None)

    experiment.local_run()
  else:
    # C major chord
    # notes = [
    #   (48, 0, .1),
    #   (60, 0, .1),
    #   (64, 0, .1),
    #   (67, 0, .1),
    # ]
    # nseq = music_pb2.NoteSequence()
    # for pitch, start_time, end_time in notes:
    #   note = nseq.notes.add()
    #   note.pitch = pitch
    #   note.velocity = 100
    #   note.start_time = start_time
    #   note.end_time = end_time
    # seq = sequence.PolyphonicSequence(nseq)
    nseq = midi_io.midi_to_sequence_proto(
        tf.gfile.FastGFile("bach/bwv853.mid").read())
    seq = sequence.PolyphonicSequence(nseq, max_start_step=50)
    for i in range(100):
      inputs, _ = codec.encode(seq)
      def _input_fn():
        return ({
            'inputs': tf.constant(np.array([inputs]), dtype=tf.float32),
            'lengths': tf.constant(np.array([inputs.shape[0]])),
        })
      predictions = estimator.predict(input_fn=_input_fn)
      codec.extend_seq_one_step_with_prediction_results(
          seq, predictions['labels'][0][-1])
    print seq.get_events()


if __name__ == "__main__":
  tf.app.run()

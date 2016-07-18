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

import random
import os
import sequence
import sys
import tensorflow as tf
import numpy as np
from magenta.lib import note_sequence_io
import one_hot_delta_codec


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('train_output', None,
                           'TFRecord to write SequenceExample protos to. '
                           'Contains training set.')
tf.app.flags.DEFINE_string('eval_output', None,
                           'TFRecord to write SequenceExample protos to. '
                           'Contains eval set. No eval set is produced if '
                           'this flag is not set.')
tf.app.flags.DEFINE_float('eval_ratio', 0.0,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_integer('max_voices', 10,
                            'The maxiumum number of voices allow in a '
                            'polyphonic sequence.')
tf.app.flags.DEFINE_integer('max_note_delta', 40,
                            'The maxiumum number of steps a voice is allowed '
                            'to change')
tf.app.flags.DEFINE_integer('max_intervoice_interval', 50,
                            'The maxiumum number of steps allowed between '
                            'voices.')

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.input:
    tf.logging.fatal('--input required')
    return
  if not FLAGS.train_output:
    tf.logging.fatal('--train_output required')
    return

  FLAGS.input = os.path.expanduser(FLAGS.input)
  FLAGS.train_output = os.path.expanduser(FLAGS.train_output)
  if FLAGS.eval_output:
    FLAGS.eval_output = os.path.expanduser(FLAGS.eval_output)

  if not os.path.exists(os.path.dirname(FLAGS.train_output)):
    os.makedirs(os.path.dirname(FLAGS.train_output))

  if FLAGS.eval_output:
    if not os.path.exists(os.path.dirname(FLAGS.eval_output)):
      os.makedirs(os.path.dirname(FLAGS.eval_output))

  reader = note_sequence_io.note_sequence_record_iterator(FLAGS.input)
  train_writer = tf.python_io.TFRecordWriter(FLAGS.train_output)
  eval_writer = (tf.python_io.TFRecordWriter(FLAGS.eval_output)
                 if FLAGS.eval_output else None)

  input_count = 0
  train_output_count = 0
  eval_output_count = 0
  tf.logging.info('Extracting polyphonic sequences...')
  for sequence_data in reader:
    tf.logging.info("Parsing data from %s" % (sequence_data.filename))
    try:
      tf.logging.info("Creating polyphonic sequence...")
      polyphonic_sequence = sequence.PolyphonicSequence(sequence_data)
      events = polyphonic_sequence.get_events()
      tf.logging.info("Got sequence of length %d with %d voices" % (
          events.shape[0], events.shape[1]))
    except (
        sequence.BadNoteException, sequence.MultipleMidiProgramsException) as e:
      tf.logging.warn("Exception while processing %s: %s" % (
        sequence_data.filename, e))
      continue
    try:
      tf.logging.info("Encoding sequence for training...")
      inputs, labels = one_hot_delta_codec.encode(
        polyphonic_sequence, FLAGS.max_voices, FLAGS.max_note_delta,
        FLAGS.max_intervoice_interval)
    except one_hot_delta_codec.EncodingException as e:
      tf.logging.warn("Exception while encoding %s: %s" % (
        sequence_data.filename, e))
      continue
    tf.logging.info("Creating sequence example...")
    sequence_example = one_hot_delta_codec.as_sequence_example(inputs, labels)
    serialized = sequence_example.SerializeToString()
    if eval_writer and random.random() < FLAGS.eval_ratio:
      eval_writer.write(serialized)
      eval_output_count += 1
    else:
      train_writer.write(serialized)
      train_output_count += 1
    input_count += 1
    tf.logging.log_every_n(
      tf.logging.INFO,
      'Extracted %d polyphonic sequences.',
      10,
      input_count)

  tf.logging.info('Done. Extracted %d polyphonic sequences.', input_count)
  tf.logging.info('Extracted %d sequences for training.', train_output_count)
  if eval_writer:
    tf.logging.info('Extracted %d sequences for evaluation.', eval_output_count)

if __name__ == "__main__":
  tf.app.run()

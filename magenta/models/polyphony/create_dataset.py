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
"""Create basic RNN dataset from NoteSequence protos.

This script will extract melodies from NoteSequence protos and save them to
TensorFlow's SequenceExample protos for input to the basic RNN model.
"""

import sequence
import logging
import sys
import tensorflow as tf

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
                          'Fraction of input to set asside for eval set. '
                          'Partition is randomly selected.')

def main(unused_argv):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  reader = note_sequence_io.note_sequence_record_iterator(sequences_file)
  input_count = 0
  for sequence_data in reader:
    polyphonic_sequence = sequence.PolyphonicSequence(sequence_data)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
  tf.app.run()

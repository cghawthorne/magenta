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
from magenta.lib import note_sequence_io

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', None,
                           'TFRecord to read NoteSequence protos from.')

def main(unused_argv):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  reader = note_sequence_io.note_sequence_record_iterator(FLAGS.input)
  input_count = 0
  for note_sequence in reader:
    polyphonic_sequence = sequence.PolyphonicSequence(note_sequence)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
  tf.app.run()

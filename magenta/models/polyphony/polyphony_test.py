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
"""Tests for polyphony."""

import numpy as np
import tensorflow as tf
import polyphony

from magenta.protobuf import music_pb2

def create_note_sequence(notes):
  sequence = music_pb2.NoteSequence()
  for pitch, start_time, end_time in notes:
    note = sequence.notes.add()
    note.pitch = pitch
    note.velocity = 100
    note.start_time = start_time
    note.end_time = end_time
  return sequence

class PolyphonyTest(tf.test.TestCase):
  def testVoiceAssignment(self):
    note_sequence = create_note_sequence([
      (50, 0, .1),
      (60, 0, .1),
      (40, .05, .15),
      (61, .15, .2),
      (51, .15, .2),
    ])
    seq = polyphony.PolyphonicSequence(note_sequence)
    expected = np.array([
        [60, 50,  -2],
        [-1, -1, 40],
        [-1, -1, -1],
        [61, 51, -1],
        [-1, -1,  -2]])

    self.assertTrue((expected == seq.get_events()).all())

if __name__ == '__main__':
    tf.test.main()

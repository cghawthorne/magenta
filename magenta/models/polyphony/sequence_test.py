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
import sequence
import test_helper

from magenta.protobuf import music_pb2

class PolyphonicSequenceTest(tf.test.TestCase):
  def testVoiceAssignment(self):
    note_sequence = test_helper.create_note_sequence([
      (50, 0, .15),
      (60, 0, .15),
      (40, .05, .2),
      (61, .15, .25),
      (51, .15, .25),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    expected = np.array([
        [-2, 50, 60],
        [40, -1, -1],
        [-1, -1, -1],
        [-1, 51, 61],
        [-2, -1, -1]])

    np.testing.assert_array_equal(expected, seq.get_events())

  def testVoiceAssignmentNonZeroStart(self):
    note_sequence = test_helper.create_note_sequence([
      (51, .15, .2),
      (61, .15, .2),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    expected = np.array([[51, 61]])

    np.testing.assert_array_equal(expected, seq.get_events())

  def testSameStopAndStartTimes(self):
    note_sequence = test_helper.create_note_sequence([
      (50, 0, .1),
      (60, 0, .1),
      (61, .1, .2),
      (51, .1, .2),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    expected = np.array([
        [50, 60],
        [-1, -1],
        [51, 61],
        [-1, -1]])

    np.testing.assert_array_equal(expected, seq.get_events())

if __name__ == '__main__':
    tf.test.main()

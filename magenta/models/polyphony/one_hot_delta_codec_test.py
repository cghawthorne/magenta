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
import one_hot_delta_codec

from magenta.protobuf import music_pb2

class OneHotDeltaCodecTest(tf.test.TestCase):
  def testEncoding(self):
    note_sequence = test_helper.create_note_sequence([
      (50, 0, .15),
      (60, 0, .15),
      (70, .05, .2),
      (61, .15, .25),
      (49, .15, .25),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)

    encoded = one_hot_delta_codec.encode(seq)

    expected = np.zeros_like(encoded)
    max_delta = 70 - 49
    delta_offset = sequence.NUM_SPECIAL_EVENTS + max_delta
    no_event = sequence.NUM_SPECIAL_EVENTS + sequence.NO_EVENT
    note_hold = sequence.NUM_SPECIAL_EVENTS + sequence.NOTE_HOLD
    expected[0][0][delta_offset + 0] = 1
    expected[0][1][delta_offset + 0] = 1
    expected[0][2][no_event] = 1
    expected[1][0][note_hold] = 1
    expected[1][1][note_hold] = 1
    expected[1][2][delta_offset + 0] = 1
    expected[2][0][note_hold] = 1
    expected[2][1][note_hold] = 1
    expected[2][2][note_hold] = 1
    expected[3][0][delta_offset - 1] = 1
    expected[3][1][delta_offset + 1] = 1
    expected[3][2][note_hold] = 1
    expected[4][0][note_hold] = 1
    expected[4][1][note_hold] = 1
    expected[4][2][no_event] = 1

    np.testing.assert_array_equal(expected, encoded)

if __name__ == '__main__':
    tf.test.main()

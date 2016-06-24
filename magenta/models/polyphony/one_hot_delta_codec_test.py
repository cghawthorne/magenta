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
    # seq.get_events() =
    # array([[50, 60, -2],
    #        [-1, -1, 70],
    #        [-1, -1, -1],
    #        [49, 61, -1],
    #        [-1, -1, -2]])

    encoded = one_hot_delta_codec.encode(seq)

    exp = np.zeros_like(encoded)
    max_delta = 70 - 49
    one_hot_delta_length = (max_delta * 2) + 1 + sequence.NUM_SPECIAL_EVENTS
    one_hot_voice_relation_length = max_delta + 1
    no_event = sequence.NUM_SPECIAL_EVENTS + sequence.NO_EVENT
    note_hold = sequence.NUM_SPECIAL_EVENTS + sequence.NOTE_HOLD

    delta_offset = sequence.NUM_SPECIAL_EVENTS + max_delta
    voices_offset = 3 * one_hot_delta_length

    # voice pairings are [(0, 1), (0, 2), (1, 2)]
    exp[0][0 * one_hot_delta_length + delta_offset + 0] = 1
    exp[0][1 * one_hot_delta_length + delta_offset + 0] = 1
    exp[0][2 * one_hot_delta_length + no_event] = 1
    exp[0][voices_offset + one_hot_voice_relation_length * 0 + 10] = 1 # [0,1]
    exp[1][0 * one_hot_delta_length + note_hold] = 1
    exp[1][1 * one_hot_delta_length + note_hold] = 1
    exp[1][2 * one_hot_delta_length + delta_offset + 0] = 1
    exp[1][voices_offset + one_hot_voice_relation_length * 0 + 10] = 1 # [0,1]
    exp[1][voices_offset + one_hot_voice_relation_length * 1 + 20] = 1 # [0,2]
    exp[1][voices_offset + one_hot_voice_relation_length * 2 + 10] = 1 # [1,2]
    exp[2][0 * one_hot_delta_length + note_hold] = 1
    exp[2][1 * one_hot_delta_length + note_hold] = 1
    exp[2][2 * one_hot_delta_length + note_hold] = 1
    exp[2][voices_offset + one_hot_voice_relation_length * 0 + 10] = 1 # [0,1]
    exp[2][voices_offset + one_hot_voice_relation_length * 1 + 20] = 1 # [0,2]
    exp[2][voices_offset + one_hot_voice_relation_length * 2 + 10] = 1 # [1,2]
    exp[3][0 * one_hot_delta_length + delta_offset - 1] = 1
    exp[3][1 * one_hot_delta_length + delta_offset + 1] = 1
    exp[3][2 * one_hot_delta_length + note_hold] = 1
    exp[3][voices_offset + one_hot_voice_relation_length * 0 + 12] = 1 # [0,1]
    exp[3][voices_offset + one_hot_voice_relation_length * 1 + 21] = 1 # [0,2]
    exp[3][voices_offset + one_hot_voice_relation_length * 2 + 9] = 1 # [1,2]
    exp[4][0 * one_hot_delta_length + note_hold] = 1
    exp[4][1 * one_hot_delta_length + note_hold] = 1
    exp[4][2 * one_hot_delta_length + no_event] = 1
    exp[4][voices_offset + one_hot_voice_relation_length * 0 + 12] = 1 # [0,1]

    np.testing.assert_array_equal(exp, encoded)

if __name__ == '__main__':
    tf.test.main()

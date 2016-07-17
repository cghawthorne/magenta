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

    max_delta = 70 - 49
    max_intervoice_interval = 25
    encoded, labels = one_hot_delta_codec.encode(
        seq, max_voices=3, max_note_delta=max_delta,
        max_intervoice_interval=max_intervoice_interval)

    exp_labels = np.zeros_like(labels)

    se = sequence.NUM_SPECIAL_EVENTS
    no_event = se + sequence.NO_EVENT
    note_hold = se + sequence.NOTE_HOLD

    exp_labels[0][0] = note_hold
    exp_labels[0][1] = note_hold
    exp_labels[0][2] = se + max_intervoice_interval + 70 - 50
    exp_labels[1][0] = note_hold
    exp_labels[1][1] = note_hold
    exp_labels[1][2] = note_hold
    exp_labels[2][0] = se + max_delta + -1
    exp_labels[2][1] = se + max_intervoice_interval + 61 - 49
    exp_labels[2][2] = note_hold
    exp_labels[3][0] = note_hold
    exp_labels[3][1] = note_hold
    exp_labels[3][2] = no_event
    exp_labels[4][0] = no_event
    exp_labels[4][1] = no_event
    exp_labels[4][2] = no_event
    np.testing.assert_array_equal(exp_labels, labels)

    exp = np.zeros_like(encoded)
    one_hot_delta_length = (max_delta * 2) + 1 + sequence.NUM_SPECIAL_EVENTS
    one_hot_voice_rel_len = 12 + max_delta + 1

    delta_offset = sequence.NUM_SPECIAL_EVENTS + max_delta
    floats_offset = 3 * one_hot_delta_length
    voices_offset = floats_offset + 3

    # voice pairings are [(0, 1), (0, 2), (1, 2)]
    exp[0][0 * one_hot_delta_length + delta_offset + 0] = 1
    exp[0][1 * one_hot_delta_length + delta_offset + 0] = 1
    exp[0][2 * one_hot_delta_length + no_event] = 1
    exp[0][floats_offset + 0] = 1 + (50/127.0)
    exp[0][floats_offset + 1] = 1 + (60/127.0)
    exp[0][voices_offset + one_hot_voice_rel_len * 0 + 10] = 1 # [0,1] oct
    exp[0][voices_offset + one_hot_voice_rel_len * 0 + 12 + 10] = 1 # [0,1]
    exp[1][0 * one_hot_delta_length + note_hold] = 1
    exp[1][1 * one_hot_delta_length + note_hold] = 1
    exp[1][2 * one_hot_delta_length + delta_offset + 0] = 1
    exp[1][floats_offset + 0] = 1 + (50/127.0)
    exp[1][floats_offset + 1] = 1 + (60/127.0)
    exp[1][floats_offset + 2] = 1 + (70/127.0)
    exp[1][voices_offset + one_hot_voice_rel_len * 0 + 10] = 1 # [0,1] oct
    exp[1][voices_offset + one_hot_voice_rel_len * 0 + 12 + 10] = 1 # [0,1]
    exp[1][voices_offset + one_hot_voice_rel_len * 1 + 8] = 1 # [0,2] oct
    exp[1][voices_offset + one_hot_voice_rel_len * 1 + 12 + 20] = 1 # [0,2]
    exp[1][voices_offset + one_hot_voice_rel_len * 2 + 10] = 1 # [1,2] oct
    exp[1][voices_offset + one_hot_voice_rel_len * 2 + 12 + 10] = 1 # [1,2]
    exp[2][0 * one_hot_delta_length + note_hold] = 1
    exp[2][1 * one_hot_delta_length + note_hold] = 1
    exp[2][2 * one_hot_delta_length + note_hold] = 1
    exp[2][floats_offset + 0] = 1 + (50/127.0)
    exp[2][floats_offset + 1] = 1 + (60/127.0)
    exp[2][floats_offset + 2] = 1 + (70/127.0)
    exp[2][voices_offset + one_hot_voice_rel_len * 0 + 10] = 1 # [0,1] oct
    exp[2][voices_offset + one_hot_voice_rel_len * 0 + 12 + 10] = 1 # [0,1]
    exp[2][voices_offset + one_hot_voice_rel_len * 1 + 8] = 1 # [0,2] oct
    exp[2][voices_offset + one_hot_voice_rel_len * 1 + 12 + 20] = 1 # [0,2]
    exp[2][voices_offset + one_hot_voice_rel_len * 2 + 10] = 1 # [1,2] oct
    exp[2][voices_offset + one_hot_voice_rel_len * 2 + 12 + 10] = 1 # [1,2]
    exp[3][0 * one_hot_delta_length + delta_offset - 1] = 1
    exp[3][1 * one_hot_delta_length + delta_offset + 1] = 1
    exp[3][2 * one_hot_delta_length + note_hold] = 1
    exp[3][floats_offset + 0] = 1 + (49/127.0)
    exp[3][floats_offset + 1] = 1 + (61/127.0)
    exp[3][floats_offset + 2] = 1 + (70/127.0)
    exp[3][voices_offset + one_hot_voice_rel_len * 0 + 0] = 1 # [0,1] oct
    exp[3][voices_offset + one_hot_voice_rel_len * 0 + 12 + 12] = 1 # [0,1]
    exp[3][voices_offset + one_hot_voice_rel_len * 1 + 9] = 1 # [0,2] oct
    exp[3][voices_offset + one_hot_voice_rel_len * 1 + 12 + 21] = 1 # [0,2]
    exp[3][voices_offset + one_hot_voice_rel_len * 2 + 9] = 1 # [1,2] oct
    exp[3][voices_offset + one_hot_voice_rel_len * 2 + 12 + 9] = 1 # [1,2]
    exp[4][0 * one_hot_delta_length + note_hold] = 1
    exp[4][1 * one_hot_delta_length + note_hold] = 1
    exp[4][2 * one_hot_delta_length + no_event] = 1
    exp[4][floats_offset + 0] = 1 + (49/127.0)
    exp[4][floats_offset + 1] = 1 + (61/127.0)
    exp[4][voices_offset + one_hot_voice_rel_len * 0 + 0] = 1 # [0,1] oct
    exp[4][voices_offset + one_hot_voice_rel_len * 0 + 12 + 12] = 1 # [0,1]

    np.testing.assert_array_equal(exp, encoded)

  def testLabelEncodingWithDelayedLowestVoiceEntry(self):
    note_sequence = test_helper.create_note_sequence([
      (50, .05, .15),
      (60, 0, .15),
      (70, .05, .2),
      (61, .15, .25),
      (49, .15, .25),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    # seq.get_events() =
    # array([[-2, 60, -2],
    #        [50, -1, 70],
    #        [-1, -1, -1],
    #        [49, 61, -1],
    #        [-1, -1, -2]])

    max_delta = 70 - 49
    max_intervoice_interval = 25
    encoded, labels = one_hot_delta_codec.encode(
        seq, max_voices=3, max_note_delta=max_delta,
        max_intervoice_interval=max_intervoice_interval)

    exp_labels = np.zeros_like(labels)

    se = sequence.NUM_SPECIAL_EVENTS
    no_event = se + sequence.NO_EVENT
    note_hold = se + sequence.NOTE_HOLD

    exp_labels[0][0] = se + max_delta + 0
    exp_labels[0][1] = note_hold
    exp_labels[0][2] = se + max_intervoice_interval + 70 - 50
    exp_labels[1][0] = note_hold
    exp_labels[1][1] = note_hold
    exp_labels[1][2] = note_hold
    exp_labels[2][0] = se + max_delta + -1
    exp_labels[2][1] = se + max_intervoice_interval + 61 - 49
    exp_labels[2][2] = note_hold
    exp_labels[3][0] = note_hold
    exp_labels[3][1] = note_hold
    exp_labels[3][2] = no_event
    exp_labels[4][0] = no_event
    exp_labels[4][1] = no_event
    exp_labels[4][2] = no_event
    np.testing.assert_array_equal(exp_labels, labels)

  def testMaxVoices(self):
    note_sequence = test_helper.create_note_sequence([
      (50, 0, .15),
      (60, 0, .15),
      (70, 0, .15),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    with self.assertRaises(one_hot_delta_codec.EncodingException):
      one_hot_delta_codec.encode(seq, max_voices=2, max_note_delta=127,
          max_intervoice_interval=100)

  def testMaxNoteDelta(self):
    note_sequence = test_helper.create_note_sequence([
      (50, 0, .15),
      (60, 0, .15),
      (80, .2, .3),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    with self.assertRaises(one_hot_delta_codec.EncodingException):
      one_hot_delta_codec.encode(seq, max_voices=3, max_note_delta=10,
          max_intervoice_interval=100)

  def testMaxIntervoiceInterval(self):
    note_sequence = test_helper.create_note_sequence([
      (50, 0, 2),
      (55, 0, 2),
      (50, 3, 4),
      (55, 3, 4),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    with self.assertRaises(one_hot_delta_codec.EncodingException):
      one_hot_delta_codec.encode(seq, max_voices=3, max_note_delta=127,
          max_intervoice_interval=4)

  def testVoicePosition(self):
    note_sequence = test_helper.create_note_sequence([
      (50, 0, .15),
      (60, 0, .15),
      (70, .05, .2),
      (61, .15, .25),
      (49, .15, .25),
      (40, 0, .25),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    encoded, labels = one_hot_delta_codec.encode(
        seq, max_voices=10, max_note_delta=127, max_intervoice_interval=100)

    # 4 voices into 10 columns. Only columns 0, 2, 5, and 9 should have data.
    self.assertTrue((labels[:,[0,3,6,9]] != 0).any())
    self.assertTrue((labels[:,[1,2,4,5,7,8]] == 0).all())

  def testVoicePositionOneVoice(self):
    note_sequence = test_helper.create_note_sequence([
      (50, 0, .15),
    ])
    seq = sequence.PolyphonicSequence(note_sequence)
    encoded, labels = one_hot_delta_codec.encode(
        seq, max_voices=10, max_note_delta=127, max_intervoice_interval=100)

    # 1 voice into 10 columns. Only column 0 should have data.
    self.assertTrue((labels[:,[0]] != 0).any())
    self.assertTrue((labels[:,[1,2,3,4,5,6,7,8,9]] == 0).all())


if __name__ == '__main__':
    tf.test.main()

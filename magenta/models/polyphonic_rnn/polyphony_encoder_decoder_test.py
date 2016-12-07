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
"""Tests for polyphony_encoder_decoder."""

# internal imports

import tensorflow as tf

from magenta.models.polyphonic_rnn import polyphony_encoder_decoder
from magenta.models.polyphonic_rnn import polyphony_lib
from magenta.models.polyphonic_rnn.polyphony_lib import PolyphonicEvent
from magenta.models.polyphonic_rnn.polyphony_lib import PolyphonicStepEvent


class PolyphonyOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = polyphony_encoder_decoder.PolyphonyOneHotEncoding()

  def testEncodeDecode(self):
    start = PolyphonicEvent(
        event_type=PolyphonicEvent.START, pitch=0)
    step_end = PolyphonicEvent(
        event_type=PolyphonicEvent.STEP_END, pitch=0)
    new_note = PolyphonicEvent(
        event_type=PolyphonicEvent.NEW_NOTE, pitch=0)
    continued_note = PolyphonicEvent(
        event_type=PolyphonicEvent.CONTINUED_NOTE, pitch=60)
    continued_max_note = PolyphonicEvent(
        event_type=PolyphonicEvent.CONTINUED_NOTE, pitch=127)

    index = self.enc.encode_event(start)
    self.assertEqual(0, index)
    event = self.enc.decode_event(index)
    self.assertEqual(start, event)

    index = self.enc.encode_event(step_end)
    self.assertEqual(2, index)
    event = self.enc.decode_event(index)
    self.assertEqual(step_end, event)

    index = self.enc.encode_event(new_note)
    self.assertEqual(3, index)
    event = self.enc.decode_event(index)
    self.assertEqual(new_note, event)

    index = self.enc.encode_event(continued_note)
    self.assertEqual(191, index)
    event = self.enc.decode_event(index)
    self.assertEqual(continued_note, event)

    index = self.enc.encode_event(continued_max_note)
    self.assertEqual(258, index)
    event = self.enc.decode_event(index)
    self.assertEqual(continued_max_note, event)


class PolyphonyStepOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = polyphony_encoder_decoder.PolyphonyStepOneHotEncoding()

  def testEncodeDecode(self):
    test_event = PolyphonicStepEvent(
        event_type=PolyphonicStepEvent.START, key=0, upcoming_end=False)
    index = self.enc.encode_event(test_event)
    self.assertEqual(0, index)
    decoded_event = self.enc.decode_event(index)
    self.assertEqual(test_event, decoded_event)

    test_event = PolyphonicStepEvent(
        event_type=PolyphonicStepEvent.START, key=1, upcoming_end=False)
    index = self.enc.encode_event(test_event)
    self.assertEqual(1, index)
    decoded_event = self.enc.decode_event(index)
    self.assertEqual(test_event, decoded_event)

    test_event = PolyphonicStepEvent(
        event_type=PolyphonicStepEvent.STEP_END, key=0, upcoming_end=False)
    index = self.enc.encode_event(test_event)
    self.assertEqual(48, index)
    decoded_event = self.enc.decode_event(index)
    self.assertEqual(test_event, decoded_event)

    test_event = PolyphonicStepEvent(
        event_type=PolyphonicStepEvent.STEP_END, key=0, upcoming_end=True)
    index = self.enc.encode_event(test_event)
    self.assertEqual(60, index)
    decoded_event = self.enc.decode_event(index)
    self.assertEqual(test_event, decoded_event)

    test_event = PolyphonicStepEvent(
        event_type=PolyphonicStepEvent.STEP_END, key=1, upcoming_end=True)
    index = self.enc.encode_event(test_event)
    self.assertEqual(61, index)
    decoded_event = self.enc.decode_event(index)
    self.assertEqual(test_event, decoded_event)

    test_event = PolyphonicStepEvent(
        event_type=PolyphonicStepEvent.NOTE, key=0, upcoming_end=False)
    index = self.enc.encode_event(test_event)
    self.assertEqual(72, index)
    decoded_event = self.enc.decode_event(index)
    self.assertEqual(test_event, decoded_event)

    test_event = PolyphonicStepEvent(
        event_type=PolyphonicStepEvent.NOTE, key=1, upcoming_end=True)
    index = self.enc.encode_event(test_event)
    self.assertEqual(85, index)
    decoded_event = self.enc.decode_event(index)
    self.assertEqual(test_event, decoded_event)


class ConditionedPolyphonyEventSequenceEncoderDecoderTest(tf.test.TestCase):

  def setUp(self):
    self.enc = polyphony_encoder_decoder.ConditionedPolyphonyEventSequenceEncoderDecoder()

  def testEncode(self):
    poly_seq = polyphony_lib.PolyphonicSequence(steps_per_quarter=1)

    pe = polyphony_lib.PolyphonicEvent
    poly_events = [
        # step 0 - C
        pe(pe.NEW_NOTE, 64),
        pe(pe.NEW_NOTE, 60),
        pe(pe.STEP_END, None),
        # step 1 - D
        pe(pe.NEW_NOTE, 69),
        pe(pe.NEW_NOTE, 66),
    ]
    for event in poly_events:
      poly_seq.append(event)

    pse = polyphony_lib.PolyphonicStepEvent
    poly_step_events = [
        pse(pse.START, 0, False),     # C
        pse(pse.STEP_END, 2, False),  # D
        pse(pse.STEP_END, 1, False),  # C#
        pse(pse.STEP_END, 0, True),
        pse(pse.END, 0, False),
    ]
    pss = polyphony_lib.PolyphonicStepSequence(
        events=poly_step_events, lookahead_steps=1)

    encoded = self.enc.events_to_input(pss, poly_seq, 0)
    self.assertEqual(1.0, encoded[72])      # NOTE, 0, False
    self.assertEqual(1.0, encoded[96 + 0])  # START

    encoded = self.enc.events_to_input(pss, poly_seq, 1)
    self.assertEqual(1.0, encoded[72])       # NOTE, 0, False
    self.assertEqual(1.0, encoded[96 + 67])  # NEW_NOTE, 64

    encoded = self.enc.events_to_input(pss, poly_seq, 2)
    self.assertEqual(1.0, encoded[72])       # NOTE, 0, False
    self.assertEqual(1.0, encoded[96 + 63])  # NEW_NOTE, 60

    encoded = self.enc.events_to_input(pss, poly_seq, 3)
    self.assertEqual(1.0, encoded[74])      # NOTE, 2, False
    self.assertEqual(1.0, encoded[96 + 2])  # STEP_END

if __name__ == '__main__':
  tf.test.main()

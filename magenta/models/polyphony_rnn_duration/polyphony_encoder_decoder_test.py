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

from magenta.models.polyphony_rnn_duration import polyphony_encoder_decoder
from magenta.models.polyphony_rnn_duration.polyphony_lib import PolyphonicEvent


class PolyphonyOneHotEncodingTest(tf.test.TestCase):

  def setUp(self):
    self.enc = polyphony_encoder_decoder.PolyphonyOneHotEncoding()

  def testEncodeDecode(self):
    start = PolyphonicEvent(
        event_type=PolyphonicEvent.START)
    end = PolyphonicEvent(
        event_type=PolyphonicEvent.END)
    step_end = PolyphonicEvent(
        event_type=PolyphonicEvent.STEP_END, duration=10)
    note_1 = PolyphonicEvent(
        event_type=PolyphonicEvent.NOTE, pitch=60, duration=63)
    note_2 = PolyphonicEvent(
        event_type=PolyphonicEvent.NOTE, pitch=10, duration=10)

    index = self.enc.encode_event(start)
    self.assertEqual(0, index)
    event = self.enc.decode_event(index)
    self.assertEqual(start, event)

    index = self.enc.encode_event(end)
    self.assertEqual(1, index)
    event = self.enc.decode_event(index)
    self.assertEqual(end, event)

    # TODO manually verify these
    index = self.enc.encode_event(step_end)
    self.assertEqual(12, index)
    event = self.enc.decode_event(index)
    self.assertEqual(step_end, event)

    index = self.enc.encode_event(note_1)
    self.assertEqual(8190, index)
    event = self.enc.decode_event(index)
    self.assertEqual(note_1, event)

    index = self.enc.encode_event(note_2)
    self.assertEqual(3, index)
    event = self.enc.decode_event(index)
    self.assertEqual(note_2, event)


if __name__ == '__main__':
  tf.test.main()

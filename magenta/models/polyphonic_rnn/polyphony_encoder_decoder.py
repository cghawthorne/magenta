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
"""Classes for converting between polyphonic input and model input/output."""

from __future__ import division
import copy

# internal imports

from magenta.models.polyphonic_rnn import polyphony_lib
from magenta.models.polyphonic_rnn.polyphony_lib import PolyphonicEvent
from magenta.models.polyphonic_rnn.polyphony_lib import PolyphonicStepEvent
from magenta.music import encoder_decoder
from magenta.music import constants

EVENT_CLASSES_WITHOUT_PITCH = [
    PolyphonicEvent.START,
    PolyphonicEvent.END,
    PolyphonicEvent.STEP_END,
]

EVENT_CLASSES_WITH_PITCH = [
    PolyphonicEvent.NEW_NOTE,
    PolyphonicEvent.CONTINUED_NOTE,
]

PITCH_CLASSES = polyphony_lib.MAX_MIDI_PITCH + 1
NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE


class PolyphonyOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for polyphonic events."""

  @property
  def num_classes(self):
    return len(EVENT_CLASSES_WITHOUT_PITCH) + (
        len(EVENT_CLASSES_WITH_PITCH) * PITCH_CLASSES)

  @property
  def default_event(self):
    return PolyphonicEvent(
        event_type=PolyphonicEvent.STEP_END, pitch=0)

  def encode_event(self, event):
    if event.event_type in EVENT_CLASSES_WITHOUT_PITCH:
      return EVENT_CLASSES_WITHOUT_PITCH.index(event.event_type)
    elif event.event_type in EVENT_CLASSES_WITH_PITCH:
      return len(EVENT_CLASSES_WITHOUT_PITCH) + (
          EVENT_CLASSES_WITH_PITCH.index(event.event_type) * PITCH_CLASSES +
          event.pitch)
    else:
      raise ValueError('Unknown event type: %s' % event.event_type)

  def decode_event(self, index):
    if index < len(EVENT_CLASSES_WITHOUT_PITCH):
      return PolyphonicEvent(
          event_type=EVENT_CLASSES_WITHOUT_PITCH[index], pitch=0)

    pitched_index = index - len(EVENT_CLASSES_WITHOUT_PITCH)
    if pitched_index < len(EVENT_CLASSES_WITH_PITCH) * PITCH_CLASSES:
      event_type = len(EVENT_CLASSES_WITHOUT_PITCH) + (
          pitched_index // PITCH_CLASSES)
      pitch = pitched_index % PITCH_CLASSES
      return PolyphonicEvent(
          event_type=event_type, pitch=pitch)

    raise ValueError('Unknown event index: %s' % index)


class PolyphonyStepOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for polyphonic step events."""

  @property
  def num_classes(self):
    return (
        4 *  # START, END, STEP_END, NOTE
        2 *  # with and without upcoming_end
        NOTES_PER_OCTAVE  # Major keys
    )

  @property
  def default_event(self):
    return PolyphonicStepEvent(
        event_type=PolyphonicStepEvent.STEP_END,
        key=PolyphonicStepEvent.UNKNOWN_KEY, upcoming_end=False)

  def encode_event(self, event):
    if event.upcoming_end:
      upcoming_end = 1
    else:
      upcoming_end = 0

    offset = event.event_type * NOTES_PER_OCTAVE * 2
    offset += NOTES_PER_OCTAVE * upcoming_end
    offset += event.key
    return offset

  def decode_event(self, index):
    offset = index
    event_type = offset // (NOTES_PER_OCTAVE * 2)
    offset = offset % (NOTES_PER_OCTAVE * 2)
    upcoming_end = bool(offset // NOTES_PER_OCTAVE)
    offset = offset % NOTES_PER_OCTAVE
    key = offset
    return PolyphonicStepEvent(
        event_type=event_type, key=key, upcoming_end=upcoming_end)


class ConditionedPolyphonyEventSequenceEncoderDecoder(
    encoder_decoder.ConditionalEventSequenceEncoderDecoder):
  def __init__(self):
    super(ConditionedPolyphonyEventSequenceEncoderDecoder, self).__init__(
        encoder_decoder.OneHotEventSequenceEncoderDecoder(
            PolyphonyStepOneHotEncoding()),
        encoder_decoder.OneHotEventSequenceEncoderDecoder(
            PolyphonyOneHotEncoding()))

  def events_to_input(self, control_events, target_events, position):
    control_events_expanded = copy.deepcopy(control_events)
    control_events_expanded.add_note_events_to_fit_polyphonic_sequence(
        target_events, position + 1)
    return (super(ConditionedPolyphonyEventSequenceEncoderDecoder, self).
     events_to_input(control_events_expanded, target_events, position))

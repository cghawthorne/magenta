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

# internal imports

from magenta.models.polyphonic_rnn import polyphony_lib
from magenta.models.polyphonic_rnn.polyphony_lib import PolyphonicEvent
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


class PolyphonyPlanningEncoderDecoder(
    encoder_decoder.EventSequenceEncoderDecoder):
  """One-hot encoding for polyphonic events.

  Includes "planning" information. Each new step marker includes information
  about what key upcoming steps will be and whether an END token is coming up.
  Also encodes the position within the current measure.
  """

  def __init__(self, lookahead_steps=16, steps_per_measure=16):
    self._lookahead_steps = lookahead_steps
    self._steps_per_measure = steps_per_measure

  @property
  def input_size(self):
    return (len(EVENT_CLASSES_WITH_PITCH) * PITCH_CLASSES +
            3 +  # START, END, STEP_END events
            NOTES_PER_OCTAVE +  # Key for upcoming steps
            1 +  # Whether an END event is coming up
            self._steps_per_measure  # Which step within the measure we're on.
            )

  @property
  def num_classes(self):
    return (len(EVENT_CLASSES_WITH_PITCH) * PITCH_CLASSES +
            3 +  # START, END, STEP_END events
            NOTES_PER_OCTAVE +  # Key for upcoming steps
            1 +  # Whether an END event is coming up
            )

  @property
  def default_event_label(self):
    # Not implemented, which is OK because this is only for graphing purposes.
    return -1.0

  def events_to_input(self, events, position):
    """Returns the input vector for the given position in the sequence.

    Indices:
      [0, 127]: New note playing at this pitch
      [128, 255]: Continued note playing at this pitch
      [256]: Sequence START
      [257]: Sequence END
      [258]: STEP_END
      [259, 270]: Key for upcoming lookahead steps
      [271]: Whether END happens within lookahead steps
      [272, 272 + self._steps_per_measure): Current step within measure.

    Args:
      events: A PolyphonicSequence object.
      position: An integer event position in the sequence.
    Returns:
      An input vector, an self.input_size length list of floats.
    """
    input_ = [0.0] * self.input_size
    offset = 0

    if event.event_type in EVENT_CLASSES_WITH_PITCH:
      input_[EVENT_CLASSES_WITH_PITCH.index(event.event_type) * PITCH_CLASSES +
             event.pitch] = 1.0
    offset += 256

    if event.event_type == PolyphonicEvent.START:
      input_[offset] = 1.0
    offset += 1

    if event.event_type == PolyphonicEvent.END:
      input_[offset] = 1.0
    offset += 1

    if event.event_type == PolyphonicEvent.STEP_END:
      input_[offset] = 1.0
    offset += 1

    input_[offset + _get_lookahead_key(events, position)] = 1.0
    offset += 12

    if _is_end_within_lookahead(events, position):
      input_[offset] = 1.0
    offset += 1

    input_[offset + _get_step_within_measure(events, position)] = 1.0

    return input_

  def _get_lookahead_key(self, events, position):
    pitches = np.zeros(constants.NOTES_PER_OCTAVE)
    step_counter = 0
    lookahead_position = position
    while (current_position < len(events) and
           step_counter < self._lookahead_steps):
      event = events[lookahead_position]
      if event.event_type is in EVENT_CLASSES_WITH_PITCH:
        pitches[event.pitch % constants.NOTES_PER_OCTAVE] += 1
      elif event.event_type == PolyphonicEvent.STEP_END:
        step_counter += 1
      lookahead_position += 1

    keys = zp.zeros(constants.NOTES_PER_OCTAVE)
    for note, count in enumerate(pitches):
      keys[constants.NOTE_KEYS[note]] += count

    return keys.argmax()

  def _is_end_within_lookahead(events, position):
    for i in range(position, len(events)):
      if events[i].event_type == PolyphonicEvent.END:
        return True
    return False

  def _get_step_within_measure(self, events, position):
    step_counter = 0
    for i in range(position + 1):
      if events[i].event_type == PolyphonicEvent.STEP_END:
        step_counter += 1
    return step_counter % self._steps_per_measure

  def events_to_label(self, events, position):
    """Returns the label for the given position in the sequence.

    Indices:
      [0, 127]: New note playing at this pitch
      [128, 255]: Continued note playing at this pitch
      [256]: Sequence START
      [257]: Sequence END
      [258]: STEP_END
      [259, 270]: Key for upcoming lookahead steps
      [271]: Whether END happens within lookahead steps

    Args:
      events: A PolyphonicSequence object.
      position: An integer event position in the sequence.
    Returns:
      An label, an integer.
    """

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

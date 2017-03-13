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

from magenta.models.polyphony_rnn_duration import polyphony_lib
from magenta.models.polyphony_rnn_duration.polyphony_lib import PolyphonicEvent
from magenta.music import encoder_decoder

EVENT_CLASSES_WITHOUT_PITCH_OR_DURATION = [
    PolyphonicEvent.START,
    PolyphonicEvent.END,
]

EVENT_CLASSES_WITH_DURATION = [
    PolyphonicEvent.STEP_END,
]

EVENT_CLASSES_WITH_PITCH_AND_DURATION = [
    PolyphonicEvent.NOTE,
]

DURATION_CLASSES = len(polyphony_lib.DURATIONS)
PITCH_CLASSES = polyphony_lib.MAX_MIDI_PITCH + 1


class PolyphonyOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding for polyphonic events."""

  @property
  def num_classes(self):
    return len(EVENT_CLASSES_WITHOUT_PITCH) + (
        len(EVENT_CLASSES_WITH_PITCH) * PITCH_CLASSES)

  @property
  def default_event(self):
    return PolyphonicEvent(
        event_type=PolyphonicEvent.STEP_END, duration=0)

  def encode_event(self, event):
    if event.event_type in EVENT_CLASSES_WITHOUT_PITCH_OR_DURATION:
      return EVENT_CLASSES_WITHOUT_PITCH_OR_DURATION.index(event.event_type)
    elif event.event_type in EVENT_CLASSES_WITH_DURATION:
      return len(EVENT_CLASSES_WITHOUT_PITCH_OR_DURATION) + (
          EVENT_CLASSES_WITH_DURATION.index(event.event_type) *
          DURATION_CLASSES + event.duration)
    elif event.event_type in EVENT_CLASSES_WITH_PITCH_AND_DURATION:
      return len(EVENT_CLASSES_WITHOUT_PITCH_OR_DURATION) + (
          len(EVENT_CLASSES_WITH_DURATION) * DURATION_CLASSES) + (
          EVENT_CLASSES_WITH_PITCH_AND_DURATION.index(event.event_type) *
          (DURATION_CLASSES + PITCH_CLASSES)) + (
              (event.duration * PITCH_CLASSES) + event.pitch)
    else:
      raise ValueError('Unknown event type: %s' % event.event_type)

  def decode_event(self, index):
    if index < len(EVENT_CLASSES_WITHOUT_PITCH_OR_DURATION):
      return PolyphonicEvent(
          event_type=EVENT_CLASSES_WITHOUT_PITCH_OR_DURATION[index])

    duration_index = index - len(EVENT_CLASSES_WITHOUT_PITCH_OR_DURATION)
    if duration_index < len(EVENT_CLASSES_WITH_DURATION) * DURATION_CLASSES:
      duration_event_type = duration_index // DURATION_CLASSES
      duration = duration_index % DURATION_CLASSES
      return PolyphonicEvent(
          event_type=EVENT_CLASSES_WITH_DURATION[duration_event_type],
          duration=duration)

    pitch_duration_index = duration_index - (
        len(EVENT_CLASSES_WITH_DURATION) * DURATION_CLASSES)
    if pitch_duration_index < len(EVENT_CLASSES_WITH_PITCH_AND_DURATION) * (
        DURATION_CLASSES * PITCH_CLASSES):
      pitch_duration_event_type = pitch_duration_index // (
          DURATION_CLASSES * PITCH_CLASSES)

      pitch_duration_index -= pitch_duration_event_type * (
          DURATION_CLASSES * PITCH_CLASSES)
      duration = pitch_duration_index // PITCH_CLASSES
      pitch = pitch_duration_index % PITCH_CLASSES
      return PolyphonicEvent(
          event_type=EVENT_CLASSES_WITH_PITCH_AND_DURATION[
              pitch_duration_event_type],
          pitch=pitch,
          duration=duration)


    raise ValueError('Unknown event index: %s' % index)

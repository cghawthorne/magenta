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
"""Utility functions for creating polyphonic training datasets.

Use extract_melodies to extract monophonic melodies from a NoteSequence proto.
"""

import logging
import math
import numpy as np
import enum
from collections import namedtuple

from magenta.protobuf import music_pb2


# Special events.
class SpecialEvents(enum.IntEnum):
  NO_EVENT = 0
  NOTE_HOLD = -1

class BadNoteException(Exception):
  pass

_Note = namedtuple('_Note', ['pitch', 'start_step', 'end_step'])

class PolyphonicSequence(object):
  """Stores a quantized stream of monophonic melody events.

  Melody is an intermediate representation that all melody models
  can use. NoteSequence proto to melody code will do work to align notes
  and extract monophonic melodies. Model specific code just needs to
  convert Melody to SequenceExample protos for TensorFlow.

  Melody implements an iterable object. Simply iterate to retrieve
  the melody events.

  Melody events are integers in range [-2, 127] (inclusive),
  where negative values are the special event events: NOTE_OFF, and NO_EVENT.
  Non-negative values [0, 127] are note-on events for that midi pitch. A note
  starts at a non-negative value (that is the pitch), and is held through
  subsequent NO_EVENT events until either another non-negative value is reached
  (even if the pitch is the same as the previous note), or a NOTE_OFF event is
  reached. A NOTE_OFF starts at least one step of silence, which continues
  through NO_EVENT events until the next non-negative value.

  NO_EVENT values are treated as default filler. Notes must be inserted
  in ascending order by start time. Note end times will be truncated if the next
  note overlaps.

  Melodies can start at any non-zero time, and are shifted left so that the bar
  containing the first note-on event is the first bar.

  Attributes:
    events: A python list of melody events which are integers. Melody events are
        described above.
    offset: When quantizing notes, this is the offset between indices in
        `events` and time steps of incoming melody events. An offset is chosen
        such that the first melody event is close to the beginning of `events`.
    steps_per_bar: Number of steps in a bar (measure) of music.
    last_on: Index of last note-on event added. This index will be within
        the range of `events`.
    last_off: Index of the NOTE_OFF event that belongs to the note-on event
        at `last_on`. This index is likely not in the range of `events` unless
        _write_all_notes was called.
  """

  def __init__(self, steps_per_second=20):
    """Construct an empty Melody.

    Args:
      steps_per_bar: How many time steps per bar of music. Melody needs to know
          about bars to skip empty bars before the first note.
    """
    # Steps x Voices
    self._events = np.zeros((0,0), dtype=int)
    self._last_notes = np.zeros((0), dtype=int)
    self._steps_per_second = steps_per_second
    self._current_step = 0
    self._current_step_notes = set()

  def _available_voice_indices():
    if self._current_step == 0:
      return []
    else:
      return (self._events[self._current_step - 1] == 0).nonzero()[0]

  def _add_note(self, note):
    """Adds the given note to the stream.

    The previous note's end step will be changed to end before this note if
    there is overlap.

    The note is not added if `start_step` is before the start step of the
    previously added note, or if `start_step` equals `end_step`.

    Args:
      pitch: Midi pitch. An integer between 0 and 127 inclusive.
      start_step: A non-zero integer step that the note begins on.
      end_step: An integer step that the note ends on. The note is considered to
          end at the onset of the end step. `end_step` must be greater than
          `start_step`.
    """
    #if not self._can_add_note(start_step):
    #  raise BadNoteException(
    #      'Given start step %d is before last on event at %d'
    #      % (start_step, self.last_on))

    # Assumes we get sorted input
    if self._current_step != note.start_step:
      # Sort self._current_step_notes into voices and add to self._events
      available_voice_indices = self._available_voice_indices()
      comb = itertools.combinations(self._current_step_notes, len(available_voice_indices))
      distance = lambda notes: np.sum(np.abs(np.array([note.pitch for n in notes]) - self._last_notes[available_voice_indices]))
      closest_voices = sorted(comb, key=distance)[0]

      # resize _events and _last_notes if needed
      if self._events.shape[1] < len(self._current_step_notes):
        self._events.resize((self._events.shape[0], len(self._current_step_notes)))
        self._last_notes.resize(len(self._current_step_notes))

      # determine array positions
      voices = closest_voices + (self._current_step_notes - set(closest_voices))
      num_new_voices = len(voices) - len(available_voice_indices)
      indices = available_voice_indices + range(self._events.shape[1] - num_new_voices, self._events.shape[1])
      for i, note in zip(indices, voices):
        print "%d: %s" % (i, note)
      
      # maintain _last_notes
      self._current_step = note.start_step

    self._current_step_notes.add(note)


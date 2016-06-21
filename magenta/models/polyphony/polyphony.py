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
import itertools
from collections import namedtuple

from magenta.protobuf import music_pb2


# Special events.
NO_EVENT = 0
NOTE_HOLD = -1

MIDI_PITCH_OFFSET = 1 # Because 0 is NO_EVENT

class BadNoteException(Exception):
  pass

class _Note(namedtuple('_Note', ['pitch', 'start_step', 'end_step'])):
  __slots__ = ()
  @property
  def pitch_with_offset(self):
    return self.pitch + MIDI_PITCH_OFFSET

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

  def _available_voice_indices(self):
    if self._events.shape[0] <= self._current_step:
      # current step is beyond any written events.
      # all voices are available
      return range(0, self._events.shape[1])
    else:
      return (self._events[self._current_step] == NO_EVENT).nonzero()[0]

  def _add_note_to_voice(self, voice, note):
    # resize _events if needed
    if self._events.shape[1] <= voice:
      self._events.resize((self._events.shape[0], voice + 1))
    if self._events.shape[0] <= note.end_step:
      self._events.resize((note.end_step + 1, self._events.shape[1]))

    self._events[note.start_step][voice] = note.pitch_with_offset
    self._events[note.start_step+1:note.end_step+1][...,voice] = NOTE_HOLD

  def _update_last_notes(self):
    # resize _last_notes if needed
    if self._last_notes.shape[0] < self._events.shape[1]:
      self._last_notes.resize(self._events.shape[1])

    for i, note in enumerate(self._events[self._current_step]):
      if note == NO_EVENT:
        # no note is active on this voice, keep the previous note
        pass
      elif note == NOTE_HOLD:
        # last note is still held
        pass
      else:
        # new note is active on this voice
        self._last_notes[i] = note

  def _voicing_distance_from_last_notes(self, voice_indices, notes):
    distance = 0
    for voice_index, note in zip(voice_indices, notes):
      if note is None:
        continue
      distance += abs(note.pitch_with_offset - self._last_notes[voice_index])
    return distance

  def _write_current_step_notes(self):
    if len(self._current_step_notes) == 0:
      return

    notes_to_assign = self._current_step_notes

    available_voice_indices = self._available_voice_indices()
    voicing = [None] * len(voices_to_assign)
    if len(available_voice_indices) > 0:
      # Pad note list with None entries so it is at least as long as
      # available_voice_indices. This lets itertools.combinations find all
      # possible assignments for us.
      padded_notes_to_assign = [notes_to_assign] + ([None] * (len(available_voice_indices) - len(notes_to_assign)))
      voicings = itertools.combinations(padded_notes_to_assign, len(available_voice_indices))
      best_voicing = sorted(
          voicings,
          key=lambda potential_voicing: self._voicing_distance_from_last_notes(
            available_voice_indices, potential_voicing)
          )[0]
      for voice_index, note in zip(available_voice_indices, best_voicing):
        notes_to_assign.remove(note)
        voicing[voice_index] = note

    # fill in other voices
    for i in range(len(voicing)):
      if voicing[i] is None:
        voicing[i] = notes_to_assign.pop()

    assert(not notes_to_assign)
    assert(None not in notes_to_assign)
    for voice, note in enumerate(voicing):
      self._add_note_to_voice(voice, note)

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
      self._write_current_step_notes()
      self._update_last_notes()
      self._current_step_notes.clear()
      self._current_step = note.start_step

    self._current_step_notes.add(note)


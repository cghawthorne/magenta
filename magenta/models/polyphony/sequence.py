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
"""Utility functions for creating polyphonic training datasets. """

import logging
import math
import numpy as np
import itertools
from collections import namedtuple

from magenta.protobuf import music_pb2


# Special events.
NOTE_HOLD = -1
NO_EVENT = -2

NUM_SPECIAL_EVENTS = 2

class BadNoteException(Exception):
  pass

class MultipleMidiProgramsException(Exception):
  pass

Note = namedtuple('Note', ['pitch', 'start_step', 'end_step'])

class PolyphonicSequence(object):
  """Stores a quantized stream of polyphonic events. """

  def __init__(self, note_sequence, steps_per_second=15, max_start_step=3000):
    """Construct a polyphonic sequence.

    Args:
      note_sequence: The NoteSequence to convert.
      steps_per_second: Controls how many steps per second are used in
          quantizing the music.
          120 bpm means 16 32nd notes per second, so 20 steps per second
          supports slightly higher resolution than 32nd notes at 120bpm.
    """
    self._max_start_step = max_start_step
    # Steps x Voices
    self._events = np.empty((0,0), dtype=int)
    self._last_notes = np.empty((0), dtype=int)
    self._steps_per_second = steps_per_second
    self._current_step = 0
    self._current_step_notes = set()

    self._add_notes(note_sequence.notes)

  def _available_voice_indices(self):
    if self._events.shape[0] <= self._current_step:
      # current step is beyond any written events.
      # all voices are available
      return range(0, self._events.shape[1])
    else:
      return (self._events[self._current_step] == NO_EVENT).nonzero()[0]

  def _write_note_to_voice(self, voice, note):
    # resize _events if needed
    if self._events.shape[1] <= voice:
      expansion = np.empty(
        (self._events.shape[0], voice - self._events.shape[1] + 1),
        dtype=int)
      expansion.fill(NO_EVENT)
      self._events = np.hstack((self._events, expansion))
    while self._events.shape[0] <= note.end_step+1:
      # Resizing is expensive, double the number of steps.
      # We'll trim it at the end.
      expansion = np.empty(
          (max(10, self._events.shape[0]), self._events.shape[1]),
          dtype=int)
      expansion.fill(NO_EVENT)
      self._events = np.vstack((self._events, expansion))

    self._events[note.start_step][voice] = note.pitch
    self._events[note.start_step+1:note.end_step+1][...,voice] = NOTE_HOLD

  def _update_last_notes(self):
    if self._events.shape[0] <= 0:
      return

    # resize _last_notes if needed
    if self._last_notes.shape[0] < self._events.shape[1]:
      self._last_notes.resize(self._events.shape[1])

    for i, note in enumerate(self._events[self._current_step]):
      if note >= 0:
        # new note is active on this voice
        self._last_notes[i] = note

  def _voicing_distance_from_last_notes(self, voice_indices, notes):
    distance = 0
    for voice_index, note in zip(voice_indices, notes):
      if note is None:
        continue
      distance += abs(note.pitch - self._last_notes[voice_index])
    return distance

  def _next_new_voice_index(self):
    return self._events.shape[1];

  def _write_current_step_notes(self):
    if len(self._current_step_notes) == 0:
      return

    notes_to_assign = self._current_step_notes

    # First try to find the best way to allocate notes into existing
    # available voices.
    available_voice_indices = self._available_voice_indices()
    if len(available_voice_indices) > 0:
      # Pad note list with None entries so it is at least as long as
      # available_voice_indices. This lets itertools.combinations find all
      # possible assignments for us.
      padded_notes_to_assign = list(notes_to_assign) + ([None] * (len(available_voice_indices) - len(notes_to_assign)))
      voicings = itertools.permutations(padded_notes_to_assign, len(available_voice_indices))
      best_voicing = sorted(
          voicings,
          key=lambda potential_voicing: self._voicing_distance_from_last_notes(
            available_voice_indices, potential_voicing)
          )[0]
      for voice_index, note in zip(available_voice_indices, best_voicing):
        if note is None:
          continue
        self._write_note_to_voice(voice_index, note)
        notes_to_assign.remove(note)

    # All the remaining voices will be new voices
    while notes_to_assign:
      self._write_note_to_voice(
          self._next_new_voice_index(),
          notes_to_assign.pop())

  def _add_note(self, note):
    """Adds the given note to the stream.

    Args:
      note: the Note to add.
    """
    assert(note.start_step >= self._current_step)

    if note.start_step > self._current_step:
      self._write_current_step_notes()
      self._update_last_notes()
      self._current_step = note.start_step

    self._current_step_notes.add(note)

  def _quantize(self, time):
    return int(round(time * self._steps_per_second))

  def _add_notes(self, notes):
    # TODO: filter out percussion notes
    active_program = None
    for note in sorted(notes, key=lambda note: note.start_time):
      # Ignore 0 velocity notes.
      if not note.velocity:
        continue

      # Do not allow notes to start or end in negative time.
      if note.start_time < 0 or note.end_time < 0:
        raise BadNoteException(
            'Got negative note time: start_time = %s, end_time = %s'
            % (note.start_time, note.end_time))

      # Allow only 1 active program
      active_program = active_program or note.program
      if active_program != note.program:
        raise MultipleMidiProgramsException(
            'Got new program %s when active program is %s'
            % (note.program, active_program))

      start_step = self._quantize(note.start_time)
      if start_step > self._max_start_step:
        break
      # make stop step 1 less than the quantized end time, to allow a new note
      # to start on the same voice at the same time, but check that we don't
      # try to stop the note before it starts.
      stop_step = max(start_step, self._quantize(note.end_time) - 1)
      self._add_note(Note(note.pitch, start_step, stop_step))


    # flush any remaining notes
    self._write_current_step_notes()

    # trim any leading silence
    while (all(self._events[0] == NO_EVENT)):
      self._events = np.delete(self._events, 0, 0)

    # trim any trailing silence
    while (all(self._events[-1] == NO_EVENT)):
      self._events = np.delete(self._events, -1, 0)

    # sort voices
    mean_notes = [
        np.mean(self._events[:,voice][self._events[:,voice] >= 0]) for
        voice in range(self._events.shape[1])]
    self._events = self._events[:, np.argsort(mean_notes)]

  def get_events(self):
    return np.copy(self._events)


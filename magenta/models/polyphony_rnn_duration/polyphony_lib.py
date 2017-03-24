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
"""Utility functions for working with polyphonic sequences."""

from __future__ import division

import collections
import copy

# internal imports

from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2

MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH
STANDARD_PPQ = constants.STANDARD_PPQ

# 120 bpm = quarter note takes 2s, 1/64th = 2/64
TICK_LENGTH = 2.0 / 64
DURATIONS = np.arange(TICK_LENGTH, 2 + TICK_LENGTH, TICK_LENGTH)

class PolyphonicEvent(object):
  """Class for storing events in a polyphonic sequence."""

  # Beginning of the sequence.
  START = 0
  # End of the sequence.
  END = 1
  # End of a step within the sequence.
  STEP_END = 2
  # Start of a new note.
  NOTE = 3

  def __init__(self, event_type, pitch=None, duration=None):
    if not (PolyphonicEvent.START <= event_type <=
            PolyphonicEvent.NOTE):
      raise ValueError('Invalid event type: %s' % event_type)
    # TODO
      #raise ValueError('Invalid data: %s for event %s' % (data, event_type))

    self.event_type = event_type
    self.pitch = pitch
    self.duration = duration

  def __repr__(self):
    return 'PolyphonicEvent(%r, %r, %r)' % (
        self.event_type, self.pitch, self.duration)

  def __eq__(self, other):
    if not isinstance(other, PolyphonicEvent):
      return False
    return (self.event_type == other.event_type and
            self.pitch == other.pitch and
            self.duration == other.duration)


class PolyphonicSequence(events_lib.EventSequence):
  """Stores a polyphonic sequence as a stream of single-note events.

  Events are PolyphonicEvent tuples that encode event type and data.
  """

  def __init__(self, note_sequence=None, start_sec=0):
    """Construct a PolyphonicSequence.

    Args:
      note_sequence: a NoteSequence proto.
      start_sec: The offset of this sequence relative to the
          beginning of the source sequence. If a note sequence is used as
          input, only notes starting after this time will be considered.
    """
    if note_sequence:
      self._events = self._from_note_sequence(note_sequence, start_sec)
    else:
      self._events = [
          PolyphonicEvent(event_type=PolyphonicEvent.START)]

    self._start_sec = start_sec

  ### Methods related to steps that are irrelevant.
  # TODO(fjord): modify interface of EventSequence so that models that don't use
  # steps can implement it more easily.

  @property
  def start_step(self):
    raise NotImplementedError()

  @property
  def end_step(self):
    raise NotImplementedError()

  @property
  def steps_per_quarter(self):
    raise NotImplementedError()

  def set_length(self, steps, from_left=False):
    raise NotImplementedError()

  ##################

  @property
  def start_sec(self):
    return self._start_sec

  @property
  def num_secs(self):
    """Returns how many steps long this sequence is.

    Does not count incomplete steps (i.e., steps that do not have a terminating
    STEP_END event).

    Returns:
      Length of the sequence in quantized steps.
    """
    secs = 0
    for event in self:
      if event.event_type == PolyphonicEvent.STEP_END:
        secs += DURATIONS[event.duration]
    return secs

  def trim_trailing_end_events(self):
    """Removes the trailing END event if present.

    Should be called before using a sequence to prime generation.
    """
    while self._events[-1].event_type == PolyphonicEvent.END:
      del self._events[-1]

  def _append_silence_steps(self, num_steps):
    """Adds steps of silence to the end of the sequence."""
    for _ in range(num_steps):
      self._events.append(
          PolyphonicEvent(event_type=PolyphonicEvent.STEP_END, pitch=None))

  def _trim_steps(self, num_steps):
    """Trims a given number of steps from the end of the sequence."""
    steps_trimmed = 0
    for i in reversed(range(len(self._events))):
      if self._events[i].event_type == PolyphonicEvent.STEP_END:
        if steps_trimmed == num_steps:
          del self._events[i + 1:]
          break
        steps_trimmed += 1
      elif i == 0:
        self._events = [
            PolyphonicEvent(event_type=PolyphonicEvent.START, pitch=None)]
        break

  def set_length_secs(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads with silence to make the
    sequence the specified length. If it is too long, it will be truncated to
    the requested length.

    Note that this will append a STEP_END event to the end of the sequence if
    there is an unfinished step.

    Args:
      steps: How many quantized steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    # TODO implement seconds-only version of this
    raise NotImplementedError()

    if from_left:
      raise NotImplementedError('from_left is not supported')

    # First remove any trailing end events.
    self.trim_trailing_end_events()
    # Then add an end step event, to close out any incomplete steps.
    self._events.append(
        PolyphonicEvent(event_type=PolyphonicEvent.STEP_END, pitch=None))
    # Then trim or pad as needed.
    if self.num_steps < steps:
      self._append_silence_steps(steps - self.num_steps)
    elif self.num_steps > steps:
      self._trim_steps(self.num_steps - steps)
    # Then add a trailing end event.
    self._events.append(
        PolyphonicEvent(event_type=PolyphonicEvent.END, pitch=None))
    assert self.num_steps == steps

  def append(self, event):
    """Appends the event to the end of the sequence.

    Args:
      event: The polyphonic event to append to the end.
    Raises:
      ValueError: If `event` is not a valid polyphonic event.
    """
    if not isinstance(event, PolyphonicEvent):
      raise ValueError('Invalid polyphonic event: %s' % event)
    self._events.append(event)

  def __len__(self):
    """How many events are in this sequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __iter__(self):
    """Return an iterator over the events in this sequence."""
    return iter(self._events)

  def __str__(self):
    strs = []
    for event in self:
      if event.event_type == PolyphonicEvent.START:
        strs.append('START')
      elif event.event_type == PolyphonicEvent.END:
        strs.append('END')
      elif event.event_type == PolyphonicEvent.STEP_END:
        strs.append('(|||, %f)' % DURATIONS[event.duration])
      elif event.event_type == PolyphonicEvent.NOTE:
        strs.append('(NOTE, %d, %f)' % (event.pitch, DURATIONS[event.duration]))
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)
    return '\n'.join(strs)

  @staticmethod
  def _from_note_sequence(note_sequence, start_secs=0):
    """Populate self with events from the given NoteSequence object.

    Sequences start with START.

    Within a step, new pitches are started with NEW_NOTE and existing
    pitches are continued with CONTINUED_NOTE. A step is ended with
    STEP_END. If an active pitch is not continued, it is considered to
    have ended.

    Sequences end with END.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start converting the sequence at this time step.
          Assumed to be the beginning of a bar.

    Returns:
      A list of events.
    """
    events = [PolyphonicEvent(event_type=PolyphonicEvent.START)]
    cur_time = start_secs
    for note in sorted(
        sorted(note_sequence.notes, key=lambda n: n.pitch, reverse=True),
        key=lambda n: n.start_time):
      if note.start_time < start_secs:
        continue
      # Advance time if needed.
      if note.start_time != cur_time:
        delta = note.start_time - cur_time
        duration_idx = np.abs(DURATIONS - delta).argmin()
        events.append(PolyphonicEvent(event_type=PolyphonicEvent.STEP_END,
                                      duration=duration_idx))
        cur_time = note.start_time
      # Add note.
      duration_idx = np.abs(
          DURATIONS - (note.end_time - note.start_time)).argmin()
      events.append(PolyphonicEvent(event_type=PolyphonicEvent.NOTE,
                                    pitch=note.pitch,
                                    duration=duration_idx))

    events.append(PolyphonicEvent(event_type=PolyphonicEvent.END, pitch=None))

    return events

  def to_sequence(self,
                  velocity=100,
                  instrument=0,
                  program=0,
                  qpm=constants.DEFAULT_QUARTERS_PER_MINUTE,
                  base_note_sequence=None):
    """Converts the PolyphonicSequence to NoteSequence proto.

    Assumes that the sequences ends with a STEP_END followed by an END event. To
    ensure this is true, call set_length before calling this method.

    Args:
      velocity: Midi velocity to give each note. Between 1 and 127 (inclusive).
      instrument: Midi instrument to give each note.
      program: Midi program to give each note.
      qpm: Quarter notes per minute (float).
      base_note_sequence: A NoteSequence to use a starting point. Must match the
          specified qpm.

    Raises:
      ValueError: if an unknown event is encountered.

    Returns:
      A NoteSequence proto.
    """
    if base_note_sequence:
      sequence = copy.deepcopy(base_note_sequence)
      if sequence.tempos[0].qpm != qpm:
        raise ValueError(
            'Supplied QPM (%d) does not match QPM of base_note_sequence (%d)'
            % (qpm, sequence.tempos[0].qpm))
    else:
      sequence = music_pb2.NoteSequence()
      sequence.tempos.add().qpm = qpm
      sequence.ticks_per_quarter = STANDARD_PPQ

    time = self.start_sec
    for i, event in enumerate(self):
      if event.event_type == PolyphonicEvent.START:
        if i != 0:
          tf.logging.debug(
              'Ignoring START marker not at beginning of sequence at position '
              '%d' % i)
      elif event.event_type == PolyphonicEvent.END:
        if i < len(self) - 1:
          tf.logging.debug(
              'Ignoring END maker before end of sequence at position %d' % i)
      elif event.event_type == PolyphonicEvent.NOTE:
        note = sequence.notes.add(
            start_time=time,
            end_time=time + DURATIONS[event.duration],
            pitch=event.pitch,
            velocity=velocity,
            instrument=instrument,
            program=program)
      elif event.event_type == PolyphonicEvent.STEP_END:
        time += DURATIONS[event.duration]
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)

    sequence.total_time = max([n.end_time for n in sequence.notes] or [0])

    return sequence


def extract_polyphonic_sequences(
    note_sequence, start_sec=0, min_secs_discard=None,
    max_secs_discard=None):
  """Extracts a polyphonic track from the given NoteSequence.

  Currently, this extracts only one polyphonic sequence from a given track.

  Args:
    note_sequence: A NoteSequence.
    start_sec: Start extracting a sequence at this second.
    min_secs_discard: Minimum length of tracks in seconds. Shorter tracks are
        discarded.
    max_secs_discard: Maximum length of tracks in seconds. Longer tracks are
        discarded.

  Returns:
    poly_seqs: A python list of PolyphonicSequence instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  """
  stats = dict([(stat_name, statistics.Counter(stat_name)) for stat_name in
                ['polyphonic_tracks_discarded_too_short',
                 'polyphonic_tracks_discarded_too_long',
                 'polyphonic_tracks_discarded_more_than_1_program']])

  # Create a histogram measuring lengths (in seconds).
  stats['polyphonic_track_lengths_in_seconds'] = statistics.Histogram(
      'polyphonic_track_lengths_in_seconds',
      [0, 1, 10, 20, 30, 40, 50, 100, 200, 500, 1000])

  num_secs = note_sequence.total_time
  if min_secs_discard is not None and num_secs < min_secs_discard:
    stats['polyphonic_tracks_discarded_too_short'].increment()
  elif max_secs_discard is not None and num_secs > max_secs_discard:
    stats['polyphonic_tracks_discarded_too_long'].increment()

  # Allow only 1 program.
  programs = set()
  for note in note_sequence.notes:
    programs.add(note.program)
  if len(programs) > 1:
    stats['polyphonic_tracks_discarded_more_than_1_program'].increment()
    return [], stats.values()

  # Translate the quantized sequence into a PolyphonicSequence.
  poly_seq = PolyphonicSequence(note_sequence, start_sec=start_sec)

  stats['polyphonic_track_lengths_in_seconds'].increment(num_secs)

  return [poly_seq], stats.values()

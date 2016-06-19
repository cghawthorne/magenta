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
      available_voices = self._available_voice_indices()
      comb = itertools.combinations(self._current_step_notes, len(available_voices))
      #broke
      distance = lambda notes: np.sum(np.abs(np.array([note.pitch for n in notes]) - self._last_notes[available_voices]))
      closest_voices = sorted(comb, key=distance)[0]
      
      # maintain _last_notes
      self._current_step = note.start_step

    self._current_step_notes.add(note)

  def _write_all_notes(self):
    """Write remaining note off event to `events`.

    This internal method makes sure that all notes being temporarily stored in
    other instance variables, namely self.last_on and self.last_off, are
    written to self.events. __iter__ and __len__ will only return what is in
    self.events, so all notes must be written there after operating on the
    events in this instance.
    """
    self.events += [NO_EVENT] * (self.last_off - len(self.events) + 1)
    self.events[self.last_off] = NOTE_OFF
    self.last_on = None
    self.last_off = None

  def _clear(self):
    """Clear `events` and last note-on/off information."""
    self.events = []
    self.last_on = None
    self.last_off = None

  def _distance_to_last_event(self, step):
    """Returns distance of the given step to the last off event.

    Args:
      step: Step to compute the distance to.

    Returns:
      Distance between step and last off event. 0 if events are the same.
      Negative if step comes before the last off event.

    Raises:
      ValueError: When the stream is empty.
    """
    if self.last_off is None:
      raise ValueError('No events in the stream')
    return step - self.offset - self.last_off

  def get_note_histogram(self):
    """Gets a histogram of the note occurrences in a melody.

    Returns:
      A list of 12 ints, one for each note value (C at index 0 through B at
      index 11). Each int is the total number of times that note occurred in
      the melody.
    """
    np_melody = np.array(self.events, dtype=int)
    return np.bincount(
        np_melody[np_melody >= MIN_MIDI_PITCH] % NOTES_PER_OCTAVE,
        minlength=NOTES_PER_OCTAVE)

  def get_major_key(self):
    """Finds the major key that this melody most likely belong to.

    Each key is matched against the pitches in the melody. The key that
    matches the most pitches is returned. If multiple keys match equally, the
    key with the lowest index is returned (where the indexes of the keys are
    C = 0 through B = 11).

    Returns:
      An int for the most likely key (C = 0 through B = 11)
    """
    note_histogram = self.get_note_histogram()
    key_histogram = np.zeros(NOTES_PER_OCTAVE)
    for note, count in enumerate(note_histogram):
      key_histogram[NOTE_KEYS[note]] += count
    return key_histogram.argmax()

  def from_notes(self, notes, bpm=120.0, gap=16, ignore_polyphonic_notes=False):
    """Populate self with an iterable of music_pb2.NoteSequence.Note.

    BEATS_PER_BAR/4 time signature is assumed.

    The given list of notes is quantized according to the given beats per minute
    and populated into self. Any existing notes in the instance are cleared.

    0 velocity notes are ignored. The melody is ended when there is a gap of
    `gap` steps or more after a note.

    If note-on events occur at the same step, this melody is cleared and an
    exception is thrown.

    Args:
      notes: Iterable of music_pb2.NoteSequence.Note
      bpm: Beats per minute. This determines the quantization step size in
          seconds. Beats are subdivided according to `steps_per_bar` given to
          the constructor.
      gap: If this many steps or more follow a note, the melody is ended.
      ignore_polyphonic_notes: If true, any notes that come before or land on
          an already added note's start step will be ignored. If false,
          PolyphonicMelodyException will be raised.

    Raises:
      PolyphonicMelodyException: If any of the notes start on the same step when
      quantized and ignore_polyphonic_notes is False.
    """
    self._clear()

    # Compute quantization steps per second.
    steps_per_second = bpm / 60.0 * self.steps_per_bar / BEATS_PER_BAR

    quantize = lambda x: int(math.ceil(x - QUANTIZE_CUTOFF))

    # Sort track by note start times.
    notes.sort(key=lambda note: note.start_time)
    for note in notes:
      # Ignore 0 velocity notes.
      if not note.velocity:
        continue

      # Quantize the start and end times of the note.
      start_step = quantize(note.start_time * steps_per_second)
      end_step = quantize(note.end_time * steps_per_second)
      if end_step == start_step:
        end_step += 1

      # Do not allow notes to start or end in negative time.
      if start_step < 0 or end_step < 0:
        raise BadNoteException(
            'Got negative note time: start_time = %s, end_time = %s'
            % (note.start_time, note.end_time))

      # If start_step comes before or lands on an already added note's start
      # step, we cannot add it. Discard the melody because it is not monophonic.
      if not self._can_add_note(start_step):
        if ignore_polyphonic_notes:
          continue
        else:
          self._clear()
          raise PolyphonicMelodyException()

      # If a gap of `gap` or more steps is found, end the melody.
      if (len(self) and
          self._distance_to_last_event(start_step) >= gap):
        break

      # Add the note-on and off events to the melody.
      self._add_note(note.pitch, start_step, end_step)

    self._write_all_notes()

  def from_event_list(self, events):
    self.events = events

  def to_sequence(self, velocity=100, instrument=0, sequence_start_time=0.0,
                  bpm=120.0):
    """Converts the Melody to Sequence proto.

    Args:
      velocity: Midi velocity to give each note. Between 1 and 127 (inclusive).
      instrument: Midi instrument to give each note.
      sequence_start_time: A time in seconds (float) that the first note in the
        sequence will land on.
      bpm: Beats per minute (float).

    Returns:
      A NoteSequence proto encoding the given melody.
    """
    seconds_per_step = 60.0 / bpm * BEATS_PER_BAR / self.steps_per_bar

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().bpm = bpm
    sequence.ticks_per_beat = STANDARD_PPQ

    current_sequence_note = None
    for step, note in enumerate(self):
      if MIN_MIDI_PITCH <= note <= MAX_MIDI_PITCH:
        # End any sustained notes.
        if current_sequence_note is not None:
          current_sequence_note.end_time = (
              step * seconds_per_step + sequence_start_time)

        # Add a note.
        current_sequence_note = sequence.notes.add()
        current_sequence_note.start_time = (
            step * seconds_per_step + sequence_start_time)
        # Give the note an end time now just to be sure it gets closed.
        current_sequence_note.end_time = (
            (step + 1) * seconds_per_step + sequence_start_time)
        current_sequence_note.pitch = note
        current_sequence_note.velocity = velocity
        current_sequence_note.instrument = instrument

      elif note == NOTE_OFF:
        # End any sustained notes.
        if current_sequence_note is not None:
          current_sequence_note.end_time = (
              step * seconds_per_step + sequence_start_time)
          current_sequence_note = None

    return sequence


  def squash(self, min_note, max_note, transpose_to_key):
    """Transpose and octave shift the notes in this Melody.

    The key center of this melody is computed with a heuristic, and the notes
    are transposed to be in the given key. The melody is also octave shifted
    to be centered in the given range. Additionally, all notes are octave
    shifted to lie within a given range.

    Args:
      min_note: Minimum pitch (inclusive) that the resulting notes will take on.
      max_note: Maximum pitch (exclusive) that the resulting notes will take on.
      transpose_to_key: The melody is transposed to be in this key. 0 = C Major.

    Returns:
      How much notes are transposed by.
    """
    melody_key = self.get_major_key()
    key_diff = transpose_to_key - melody_key
    midi_notes = [note for note in self.events
                  if MIN_MIDI_PITCH <= note <= MAX_MIDI_PITCH]
    if not midi_notes:
      return
    melody_min_note = min(midi_notes)
    melody_max_note = max(midi_notes)
    melody_center = (melody_min_note + melody_max_note) / 2
    target_center = (min_note + max_note - 1) / 2
    center_diff = target_center - (melody_center + key_diff)
    transpose_amount = (
        key_diff +
        NOTES_PER_OCTAVE * int(round(center_diff / float(NOTES_PER_OCTAVE))))
    for i in xrange(len(self.events)):
      # Transpose MIDI pitches. Special events below MIN_MIDI_PITCH are not changed.
      if self.events[i] >= MIN_MIDI_PITCH:
        self.events[i] += transpose_amount
        if self.events[i] < min_note:
          self.events[i] = (
              min_note + (self.events[i] - min_note) % NOTES_PER_OCTAVE)
        elif self.events[i] >= max_note:
          self.events[i] = (max_note - NOTES_PER_OCTAVE +
                            (self.events[i] - max_note) % NOTES_PER_OCTAVE)

    return transpose_amount


def extract_melodies(sequence, steps_per_beat=4, min_bars=7,
                     min_unique_pitches=5):
  """Extracts a list of melodies from the given NoteSequence proto.

  A time signature of BEATS_PER_BAR is assumed for each sequence. If the
  sequence has an incompatable time signature, like 3/4, 5/4, etc, then
  the time signature is ignored and BEATS_PER_BAR/4 time is assumed.

  Once a note-on event in a track is encountered, a melody begins. Once a
  gap of silence since the last note-off event of a bar length or more is
  encountered, or the end of the track is reached, that melody is ended. Only
  the first melody of each track is used (this reduces the number of repeated
  melodies that may come from repeated choruses or verses, but may also cause
  unique non-first melodies, such as bridges and outros, to be missed, so maybe
  this should be changed).

  The melody is then checked for validity. The melody is only used if it is
  at least `min_bars` bars long, and has at least `min_unique_pitches` unique
  notes (preventing melodies that only repeat a few notes, such as those found
  in some accompaniment tracks, from being used).

  After scanning each instrument track in the NoteSequence, a list of all the valid
  melodies is returned.

  Args:
    sequence: A NoteSequence proto containing notes.
    steps_per_beat: How many subdivisions of each beat. BEATS_PER_BAR/4 time is
        assumed, so steps per bar is equal to
        `BEATS_PER_BAR` * `steps_per_beat`.
    min_bars: Minimum length of melodies in number of bars. Shorter melodies are
        discarded.
    min_unique_pitches: Minimum number of unique notes with octave equivalence.
        Melodies with too few unique notes are discarded.

  Returns:
    A python list of Melody instances.
  """

  # Assume bars contain 4 beats, or quarter notes.
  steps_per_bar = steps_per_beat * 4

  # Beats per minute is stored in the tempo change event. If there is no tempo
  # then assume 120 bpm per the MIDI standard.
  bpm = (sequence.tempos[0].bpm if len(sequence.tempos)
         else DEFAULT_BEATS_PER_MINUTE)

  # Group note messages into tracks.
  tracks = {}
  for note in sequence.notes:
    if note.instrument not in tracks:
      tracks[note.instrument] = []
    tracks[note.instrument].append(note)

  melodies = []
  for track in tracks.values():
    melody = Melody(steps_per_bar)

    # Quantize the track into a Melody object.
    # If any notes start at the same time, only one is kept.
    melody.from_notes(track, bpm=bpm, gap=steps_per_bar,
                      ignore_polyphonic_notes=True)

    # Require a certain melody length.
    if len(melody) - 1 < steps_per_bar * min_bars:
      logging.debug('melody too short')
      continue

    # Require a certain number of unique pitches.
    note_histogram = melody.get_note_histogram()
    unique_pitches = np.count_nonzero(note_histogram)
    if unique_pitches < min_unique_pitches:
      logging.debug('melody too simple')
      continue

    melodies.append(melody)

  return melodies


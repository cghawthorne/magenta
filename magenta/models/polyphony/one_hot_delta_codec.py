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
"""Converts PolyphonicSequence objects into input features and labels."""

import sequence
import numpy as np
import itertools
import tensorflow as tf

DEFAULT_MAX_VOICES = 4
DEFAULT_MIN_VOICES = 4
DEFAULT_MAX_NOTE_DELTA = 20
DEFAULT_MAX_INTERVOICE_INTERVAL = 40

class EncodingException(Exception):
  pass


class PolyphonyCodec:
  def __init__(
      self,
      max_voices=DEFAULT_MAX_VOICES,
      min_voices=DEFAULT_MIN_VOICES,
      max_note_delta=DEFAULT_MAX_NOTE_DELTA,
      max_intervoice_interval=DEFAULT_MAX_INTERVOICE_INTERVAL):
    self._max_voices = max_voices
    self._min_voices = min_voices
    self._max_note_delta = max_note_delta
    self._max_intervoice_interval = max_intervoice_interval

    self._one_hot_delta_length = (
        (self._max_note_delta * 2) + 1 + sequence.NUM_SPECIAL_EVENTS)
    self._voice_pairings = list(
        itertools.combinations(range(self._max_voices), 2))
    self._one_hot_voice_relation_length = 12 + self._max_intervoice_interval + 1
    self._pitch_floats_length = self._max_voices
    self._one_hot_length = (
        (self._one_hot_delta_length * self._max_voices) +
        self._pitch_floats_length +
        (self._one_hot_voice_relation_length * len(self._voice_pairings)))

  @property
  def input_size(self):
    return self._one_hot_length

  @property
  def labels_per_example(self):
    return self._max_voices

  @property
  def classes_per_label(self):
    lowest_voice_classes = (
        sequence.NUM_SPECIAL_EVENTS + (self._max_note_delta * 2) + 1)
    other_voice_classes = (
        sequence.NUM_SPECIAL_EVENTS + (self._max_intervoice_interval * 2) + 1)
    return [lowest_voice_classes] + (
        [other_voice_classes] * (self._max_voices -1))

  @property
  def num_classes(self):
    return sum(self.classes_per_label)

  @property
  def max_note_delta(self):
    return self._max_note_delta

  @property
  def max_intervoice_interval(self):
    return self._max_intervoice_interval

  def encode(self, polyphonic_sequence):
    seq = polyphonic_sequence.get_events()

    if seq.shape[1] > self._max_voices:
      raise EncodingException("Too many voices: %d > max of %d" %
                              (seq.shape[1], self._max_voices))

    if seq.shape[1] < self._min_voices:
      raise EncodingException("Too few voices: %d < min of %d" %
                              (seq.shape[1], self._min_voices))

    inputs = np.zeros((seq.shape[0], self._one_hot_length), dtype=float)
    # labels for upper voices are how high above or below the lowest voice they
    # are or a special event (offset by NUM_SPECIAL_EVENTS). if needed, look ahead
    # to find out what the first bass note will be.
    # number space is NUM_SPECIAL_EVENTS, max_intervoice_interval (below bass),
    # same as bass, max_intervoice_interval (above bass)
    # label for the lowest voice is how much it moved up or down or special event.
    # number space is NUM_SPECIAL_EVENTS, max_note_delta (downward movement), no
    # movement, max_note_delta (upward movement)
    labels = np.zeros((seq.shape[0], self._max_voices), dtype=int)

    first_note_lowest_voice = seq[np.where(seq[:,[0]] >= 0)[0][0]][0]
    last_notes = [None] * self._max_voices
    active_notes = [None] * self._max_voices
    for step in range(seq.shape[0]):
      for seq_voice, pitch in enumerate(seq[step]):
        voice = sequence.remap_voice_index(
            seq.shape[1], self._max_voices, seq_voice)

        if pitch == sequence.NO_EVENT:
          active_notes[voice] = None
        elif pitch >= 0:
          active_notes[voice] = pitch

        offset = voice * self._one_hot_delta_length
        if pitch < 0:
          inputs[step][offset + pitch + sequence.NUM_SPECIAL_EVENTS] = 1

          # Update labels
          if step > 0:
            labels[step - 1][voice] = pitch + sequence.NUM_SPECIAL_EVENTS
        else:
          delta = 0
          if last_notes[voice] is not None:
            delta = pitch - last_notes[voice]

          if delta > self._max_note_delta:
            raise EncodingException("Note delta too great: %d > max of %d" %
                                    (delta, self._max_note_delta))

          last_notes[voice] = pitch
          one_hot_delta = self._max_note_delta + delta
          inputs[step][offset + one_hot_delta + sequence.NUM_SPECIAL_EVENTS] = 1

          # Update labels
          if step > 0:
            # Lowest voice is delta, other voices are steps above or below
            # lowest note
            if voice == 0:
              labels[step - 1][voice] = (
                  sequence.NUM_SPECIAL_EVENTS + self._max_note_delta + delta)
            else:
              current_lowest_voice_note = last_notes[0]
              if current_lowest_voice_note is None:
                current_lowest_voice_note = first_note_lowest_voice
              interval = pitch - current_lowest_voice_note
              if interval > self._max_intervoice_interval:
                raise EncodingException(
                    "Intervoice interval too great: %d > max of %d" %
                    (interval, self._max_intervoice_interval))
              labels[step - 1][voice] = (sequence.NUM_SPECIAL_EVENTS +
                  self._max_intervoice_interval + interval)
        # Add floating point pitch information
        if pitch >= 0 or pitch == sequence.NOTE_HOLD:
          inputs[step][(self._max_voices * self._one_hot_delta_length) + voice] = 1 + (
              last_notes[voice] / 127.0)
        else:
          # No active pitch, so leave it 0
          pass
      for i, voice_pair in enumerate(self._voice_pairings):
        if not active_notes[voice_pair[0]] or not active_notes[voice_pair[1]]:
          continue
        distance = abs(active_notes[voice_pair[0]] - active_notes[voice_pair[1]])
        if distance > self._max_intervoice_interval:
          raise EncodingException(
              "Intervoice interval too great: %d > max of %d" %
              (distance, self._max_intervoice_interval))
        offset = ((self._max_voices * self._one_hot_delta_length) + self._pitch_floats_length +
            (i * self._one_hot_voice_relation_length))
        # distance ignoring octaves
        inputs[step][offset + (distance % 12)] = 1
        # absolute distance
        inputs[step][offset + 12 + distance] = 1

    return inputs, labels

  def as_sequence_example(self, inputs, labels):
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs]
    label_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=label))
        for label in labels]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)

  def extend_seq_one_step_with_prediction_results(self, seq, labels):
    new_step = np.empty(len(labels), dtype=int)
    events = seq.get_events()
    last_note_lowest_voice = events[np.where(events[:,[0]] >= 0)[0][-1]][0]
    if labels[0] < sequence.NUM_SPECIAL_EVENTS:
      new_step[0] = labels[0] - sequence.NUM_SPECIAL_EVENTS
    else:
      new_step[0] = last_note_lowest_voice + (
          labels[0] - sequence.NUM_SPECIAL_EVENTS - self._max_note_delta)
      last_note_lowest_voice = new_step[0]

    for voice in range(1, len(labels)):
      if labels[voice] < sequence.NUM_SPECIAL_EVENTS:
        new_step[voice] = labels[voice] - sequence.NUM_SPECIAL_EVENTS
      else:
        new_step[voice] = last_note_lowest_voice + (
            labels[voice] - sequence.NUM_SPECIAL_EVENTS -
            self._max_intervoice_interval)

    seq.extend_one_step(new_step)

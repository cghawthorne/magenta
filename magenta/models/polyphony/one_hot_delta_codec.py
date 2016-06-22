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

def encode(polyphonic_sequence):
  seq = polyphonic_sequence.get_events()

  # TODO: generalize to work with inputs that have different high/low notes
  seq_without_special_events = seq[seq >= 0]
  # TODO: could be more clever about calculating max_delta so we actually find
  # what the largest change between notes is, resulting in a probably-smaller
  # one-hot array.
  max_delta = (
      seq_without_special_events.max() - seq_without_special_events.min())
  one_hot_length = (max_delta * 2) + 1 + sequence.NUM_SPECIAL_EVENTS

  # TODO: generalize to work with inputs that have different numbers of voices
  inputs = np.empty(
      (seq.shape[0], seq.shape[1], one_hot_length),
      dtype=float)
  inputs.fill(sequence.NO_EVENT + sequence.NUM_SPECIAL_EVENTS)

  last_notes = [None] * seq.shape[1]
  for i in range(seq.shape[0]):
    for voice, pitch in enumerate(seq[i]):
      if pitch < 0:
        inputs[i][voice][pitch + sequence.NUM_SPECIAL_EVENTS] = 1
      else:
        delta = 0
        if last_notes[voice] is not None:
          delta = pitch - last_notes[voice]

        last_notes[voice] = pitch
        inputs[i][voice][max_delta + delta + sequence.NUM_SPECIAL_EVENTS] = 1

  # TODO: how to generate multiple labels per step?
  return inputs 

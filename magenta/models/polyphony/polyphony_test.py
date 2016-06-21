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
"""Tests for polyphony."""

import numpy as np
import tensorflow as tf
import polyphony

class PolyphonyTest(tf.test.TestCase):

  def testVoiceAssignment(self):
    seq = polyphony.PolyphonicSequence()
    notes = [
        polyphony.Note(50,0,2),
        polyphony.Note(60,0,2),
        polyphony.Note(40,1,3),
        polyphony.Note(61,3,4),
        polyphony.Note(51,3,4),
    ]
    seq.add_notes(notes)
    expected = np.array([
        [61, 51,  0],
        [-1, -1, 41],
        [-1, -1, -1],
        [62, 52, -1],
        [-1, -1,  0]])

    self.assertTrue((expected == seq.get_events()).all())

if __name__ == '__main__':
    tf.test.main()

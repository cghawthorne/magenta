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

import tensorflow as tf
import polyphony

class PolyphonyTest(tf.test.TestCase):

  def testPolyphony(self):
    seq = polyphony.PolyphonicSequence()
    n1 = polyphony._Note(50,0,5)
    n2 = polyphony._Note(60,0,5)
    n3 = polyphony._Note(61,6,10)
    n4 = polyphony._Note(51,6,10)
    seq._add_note(n1)
    seq._add_note(n2)
    seq._add_note(n3)
    seq._write_current_step_notes()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    tf.test.main()

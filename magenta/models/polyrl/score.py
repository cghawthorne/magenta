import numpy as np


class CursorActions:
  MEASURE_FORWARD = 0
  MEASURE_BACKWARD = 1
  TICK_FORWARD = 2
  TICK_BACKWARD = 3
  VOICE_UP = 4
  VOICE_DOWN = 5

class Score(object):
  def __init__(self):
    # 4 measures, 8 ticks (4 beats, 8th note resolution, 4 voices)
    # Value is a combination of:
    #   midi pitch + 1 (first 8 bits, 0 is considered empty)
    #   duration in ticks (second 8 bits)
    self.score = np.zeros(
        (4, 8, 4), dtype=[('pitch', np.int8), ('duration', np.int8)])

    # measure, tick (within measure), voice
    self.cursor = np.array([0, 0, 0], dtype=int)

  def move_cursor(self, action):
    if action == CursorActions.MEASURE_FORWARD:
      if self.cursor[0] < self.score.shape[0] - 1:
        self.cursor[0] += 1
        # also reset tick position
        self.cursor[1] = 0
      else:
        print 'cannot move measure forward'
    elif action == CursorActions.MEASURE_BACKWARD:
      if self.cursor[0] > 1:
        self.cursor[0] -= 1
        # also reset tick position
        self.cursor[1] = 0
      else:
        print 'cannot move measure backward'
    elif action == CursorActions.TICK_FORWARD:
      if self.cursor[1] < self.score.shape[1] - 1:
        self.cursor[1] += 1
      else:
        print 'cannot move tick forward'
    elif action == CursorActions.TICK_BACKWARD:
      if self.cursor[1] > 1:
        self.cursor[1] -= 1
      else:
        print 'cannot move tick backward'
    elif action == CursorActions.VOICE_UP:
      if self.cursor[2] < self.score.shape[2] - 1:
        self.cursor[2] += 1
      else:
        print 'cannot move voice up'
    elif action == CursorActions.VOICE_DOWN:
      if self.cursor[2] > 1:
        self.cursor[2] -= 1
      else:
        print 'cannot move voice down'

  def insert_note_at_cursor(self, pitch, duration):
    max_duration = self.score.shape[1] - self.cursor[1]
    if duration > max_duration:
      print 'duration too long'
      return
    if duration <= 0:
      print 'duration too short'
      return
    if pitch <= 0:
      print 'pitch too low'
    if pitch > 127:
      print 'pitch too high'

    current = self.score[tuple(cursor)]
    if np.array_equal(current, [0, 0]):
      self.score[cursor[0], cursor[1], cursor[2]] = (pitch, duration)
      # mark the rest of the duration as in use
      self.score[cursor[0], cursor[1] + 1:cursor[1] + duration, cursor[2]] = (
          -1, -1)
    else:
      print 'cursor position already has note'
      return

  def delete_note_at_cursor(self):
    current = self.score[tuple(cursor)]
    if current[0] == 0 and current[1] == 0:
      print 'no note in current position'
      return
    elif np.array_equal(current, [-1, -1]):
      print 'in the middle of a note'
    else:
      self.score[cursor[0], cursor[1]:cursor[1] + current[1], cursor[2]] = (
          0, 0)

import numpy as np

# 4 measures, 8 ticks (4 beats, 8th note resolution, 4 voices)
# Value is a combination of:
#   midi pitch + 1 (first 8 bits, 0 is considered empty)
#   duration in ticks (second 8 bits)
score = np.zeros((4, 8, 4), dtype=[('pitch', np.int8), ('duration', np.int8)])

# measure, tick (within measure), voice
cursor = np.array([0, 0, 0], dtype=int)

class CursorActions:
  MEASURE_FORWARD = 0
  MEASURE_BACKWARD = 1
  TICK_FORWARD = 2
  TICK_BACKWARD = 3
  VOICE_UP = 4
  VOICE_DOWN = 5
  
def move_cursor(action):
  if action == CursorActions.MEASURE_FORWARD:
    if cursor[0] < score.shape[0] - 1:
      cursor[0] += 1
      # also reset tick position
      cursor[1] = 0
    else:
      print 'cannot move measure forward'
  elif action == CursorActions.MEASURE_BACKWARD:
    if cursor[0] > 1:
      cursor[0] -= 1
      # also reset tick position
      cursor[1] = 0
    else:
      print 'cannot move measure backward'
  elif action == CursorActions.TICK_FORWARD:
    if cursor[1] < score.shape[1] - 1:
      cursor[1] += 1
    else:
      print 'cannot move tick forward'
  elif action == CursorActions.TICK_BACKWARD:
    if cursor[1] > 1:
      cursor[1] -= 1
    else:
      print 'cannot move tick backward'
  elif action == CursorActions.VOICE_UP:
    if cursor[2] < score.shape[2] - 1:
      cursor[2] += 1
    else:
      print 'cannot move voice up'
  elif action == CursorActions.VOICE_DOWN:
    if cursor[2] > 1:
      cursor[2] -= 1
    else:
      print 'cannot move voice down'

def insert_note_at_cursor(pitch, duration):
  max_duration = score.shape[1] - cursor[1]
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

  current = score[cursor[0], cursor[1], cursor[2]]
  if current[0] == 0 and current[1] == 0:
    score[cursor[0], cursor[1], cursor[2]] = (pitch, duration)
    # mark the rest of the duration as in use
    score[cursor[0], cursor[1]+1:cursor[1]+duration, cursor[2]] = (-1, -1)
  else:
    print 'cursor position already has note'
    return

def delete_note_at_cursor():
  current = score[cursor[0], cursor[1], cursor[2]]
  if current[0] == 0 and current[1] == 0:
    print 'no note in current position'
    return
  elif current[0] == -1 and current[1] == -1:
    print 'in the middle of a note'
  else:
    score[cursor[0], cursor[1]:cursor[1]+current[1], cursor[2]] = (0, 0)
    


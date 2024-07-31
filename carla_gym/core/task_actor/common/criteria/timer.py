import numpy as np

class Timer():

    def __init__(self, max_time=30.0):
        self._max_time = max_time
        self._time_last_valid_state = None
    
    def turn_on(self, timestamp):
        self._time_last_valid_state = timestamp['relative_simulation_time']
    
    def turn_off(self):
        self._time_last_valid_state = None

    def tick(self, timestamp):
        info = None
        if self._time_last_valid_state is not None:
            if (timestamp['relative_simulation_time'] - self._time_last_valid_state) > self._max_time:
                # Time has run out!
                info = {
                    'step': timestamp['step'],
                    'simulation_time': timestamp['relative_simulation_time'],
                }
        return info

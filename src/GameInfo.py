import time

class GameInfo:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level_start_time = time.time()
        self.score = 0

    def get_level_time(self):
        return round(time.time() - self.level_start_time, 1)
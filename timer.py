
import time

class Timer(object):
    timers = {}
    def __init__(self, name, print_=False):
        self.start = self.end = 0
        self.name = name
        self.print_ = print_

    @classmethod
    def new(cls, name, print_=False):
        cls.timers[name] = Timer(name, print_)
        return cls.timers[name]

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        if self.print_:
            print '%s: %.6fs' % (self.name, self.end - self.start)

    @classmethod
    def get(cls, name):
        return ((cls.timers[name].end - cls.timers[name].start)
                if name in cls.timers else 0)

    @classmethod
    def reset(cls):
        cls.timers = {}

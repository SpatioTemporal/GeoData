

from time import process_time

class stopwatch(object):
    timestamps = []
    names      = {}
    _timer     = process_time

    def __init__(self,timer=None):
        if timer is not None:
            self._timer = timer
        self.stamp("Instantiated")
        return

    def stamp(self,name):
        self.names[name] = len(self.timestamps)
        self.timestamps.append(self._timer())
        return self.timestamps[-1]

    def delta(self,name1=None,name2=None):
        if name1 is None and name2 is None:
            return self.timestamps[-1] - self.timestamps[0]
        if name2 is None:
            return self.timestamps[-1] -  self.timestamps[names[name1]]
        if name1 is None:
            return self.timestamps[name2] -  self.timestamps[0]
        return self.timestamps[names[name2]] - self.timestamps[names[name1]]

    def current(self):
        return self._timer()

    def delta_since_start(self):
        return self.current() - self.timestamps[0]

    def report_string(self,name1,name2,message=None):
        if message is None:
            message = "Time between %s and %s is "%(name1,name2)
        return message+"%f"%self.delta(name1,name2)



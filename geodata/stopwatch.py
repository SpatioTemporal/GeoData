

from time import process_time

class stopwatch(object):
    timestamps = []
    names      = {}
    iteration  = {}
    _timer     = process_time

    def __init__(self,timer=None):
        if timer is not None:
            self._timer = timer
        self.stamp("Instantiation")
        return

    def add_name(self,name):
        if name not in self.names.keys():
            self.iteration[name] = 0
            self.names[name] = len(self.timestamps)
        else:
            self.iteration[name] = self.iteration[name] + 1
            self.names["%s(%i)"%(name,self.iteration[name])] = len(self.timestamps)

    def stamp(self,name=None):
        if name is None:
            name = "stamp-%s"%len(self.names.keys())
        self.add_name(name)
        self.timestamps.append(self._timer())
        return self.timestamps[-1]

    def delta(self,name1=None,name2=None):
        if name1 is None and name2 is None:
            return self.timestamps[-1] - self.timestamps[0]
        if name2 is None:
            return self.timestamps[-1] -  self.timestamps[self.names[name1]]
        if name1 is None:
            return self.timestamps[name2] -  self.timestamps[0]
        return self.timestamps[self.names[name2]] - self.timestamps[self.names[name1]]

    def current(self):
        return self._timer()

    def delta_since_start(self):
        return self.current() - self.timestamps[0]

    def report_string(self,name1,name2,message=None):
        if message is None:
            message = "'%s','%s',"%(name1,name2)
        return message+"%06e"%self.delta(name1,name2)

    def report_all(self,prefix=""):
        ret_str = ""
        keys = list(self.names.keys())
        for i in range(len(self.timestamps)):
            ret_str=ret_str+prefix+self.report_string("Instantiation",keys[i])+"\n"
        return ret_str

global sw_timer
sw_timer = stopwatch()


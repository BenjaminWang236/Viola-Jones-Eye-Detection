import datetime
import string
from datetime import datetime, timedelta
from string import Template

class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    d["F"] = '{:03d}'.format(tdelta.microseconds)[:-3]
    t = DeltaTemplate(fmt)
    return t.substitute(**d)
# https://github.com/lukasheinrich/pylhe

from xml.etree import ElementTree


class LHEEvent(object):
    def __init__(self, eventinfo, particles):
        self.eventinfo = eventinfo
        self.particles = particles
        for p in self.particles:
            p.event = self


class LHEEventInfo(object):
    fieldnames = ['nparticles', 'pid', 'weight', 'scale', 'aqed', 'aqcd']

    def __init__(self, **kwargs):
        if not set(kwargs.keys()) == set(self.fieldnames):
            raise RuntimeError
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def fromstring(cls, string):
        return cls(**dict(zip(cls.fieldnames, map(float, string.split()))))


class LHEParticle(object):
    fieldnames = ['id', 'status', 'mother1', 'mother2', 'color1', 'color2', 'px', 'py', 'pz', 'e', 'm', 'lifetime', 'spin']

    def __init__(self, **kwargs):
        if not set(kwargs.keys()) == set(self.fieldnames):
            raise RuntimeError
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def fromstring(cls, string):
        obj = cls(**dict(zip(cls.fieldnames, map(float, string.split()))))
        return obj


def read_lhe(thefile):
    try:
        for event, element in ElementTree.iterparse(thefile, events=['end']):
            if element.tag == 'event':
                data = element.text.split('\n')[1:-1]
                eventdata, particles = data[0], data[1:]
                eventinfo = LHEEventInfo.fromstring(eventdata)
                particle_objs = []
                for p in particles:
                    particle_objs += [LHEParticle.fromstring(p)]
                yield LHEEvent(eventinfo, particle_objs)

    except ElementTree.ParseError:
        print("WARNING. Parse Error.")
        return

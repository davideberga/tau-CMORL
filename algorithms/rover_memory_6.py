from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                      'reward', 'cost_charger', 'cost_avoid', 'cost_battery'))

class RoverMemory6(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

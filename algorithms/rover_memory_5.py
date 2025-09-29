from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward', 'reward_goal', 'reward_charger', 'cost_avoid', 'cost_battery'))

class RoverMemory5(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

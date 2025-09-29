import numpy as np
from collections import defaultdict

class RewardNormalizer:
    def __init__(self, keys, clip_value=None, epsilon=1e-8):
        self.keys = keys
        self.clip_value = clip_value
        self.epsilon = epsilon
        
        # Running stats
        self.counts = defaultdict(int)
        self.means = defaultdict(float)
        self.M2 = defaultdict(float)  # For variance
    
    def update(self, key, value):
        """Update running mean & variance using Welford's algorithm."""
        self.counts[key] += 1
        delta = value - self.means[key]
        self.means[key] += delta / self.counts[key]
        delta2 = value - self.means[key]
        self.M2[key] += delta * delta2
    
    def std(self, key):
        if self.counts[key] < 2:
            return 1.0
        return np.sqrt(self.M2[key] / (self.counts[key] - 1))
    
    def normalize(self, key, value):
        mean = self.means[key]
        std = self.std(key)
        normalized = (value - mean) / (std + self.epsilon)
        if self.clip_value is not None:
            normalized = np.clip(normalized, -self.clip_value, self.clip_value)
        return normalized
    
    def __call__(self, rewards_dict, update=True):
        """Normalize multiple rewards at once."""
        normalized = {}
        for key in self.keys:
            value = rewards_dict[key]
            if update:
                self.update(key, value)
            normalized[key] = self.normalize(key, value)
        return normalized
 
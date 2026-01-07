import math
import numpy as np
import randomhash

class ParallelHyperLogLog:
    def __init__(self, b, num_hashes, random_hash_functions: randomhash.RandomHashFamily = None):
        self.b = b
        self.m = 1 << b  # m = 2^b
        self.num_hashes = num_hashes
        
        # N x m matrix
        self.M = [[0] * self.m for _ in range(num_hashes)]
        
        # Alpha from HLL paper
        if self.m == 16: self.alpha = 0.673
        elif self.m == 32: self.alpha = 0.697
        elif self.m == 64: self.alpha = 0.709
        else: self.alpha = 0.7213 / (1 + 1.079 / self.m)

        try:
            if random_hash_functions is not None:
                self.rf = random_hash_functions
            else:
                self.rf = randomhash.RandomHashFamily(count=num_hashes)
        except ImportError:
            raise ImportError("randomhash library not found. Please install it or implement a custom hash family.")

    def get_position_first_one(self, w):
        """Position of the leftmost 1-bit."""
        if w == 0: return 32 - self.b + 1
        rank = 1
        while (w & 1) == 0:
            w >>= 1
            rank += 1
        return rank

    def add(self, item):
        hash_values = self.rf.hashes(item)
        for k in range(self.num_hashes):
            x = hash_values[k] & 0xFFFFFFFF 
            
            # Bucketing using b
            j = x >> (32 - self.b)
            
            # Zero-run counting on remainder
            w = x & ((1 << (32 - self.b)) - 1)
            rho = self.get_position_first_one(w)
            
            # Update register
            self.M[k][j] = max(self.M[k][j], rho)

    def count(self):
        results = []
        for k in range(self.num_hashes):
            registers = self.M[k]
            sum_inv = sum(2.0 ** (-val) for val in registers)
            E = self.alpha * (self.m ** 2) / sum_inv
            if E <= 2.5 * self.m:
                V = registers.count(0)
                if V > 0: E = self.m * math.log(self.m / V)
            
            results.append(int(E))
        return results
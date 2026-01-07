import heapq
import randomhash

class ParallelRecordinality:
    def __init__(self, k, num_hashes, random_hash_functions: randomhash.RandomHashFamily = None):
        self.k = k
        self.num_hashes = num_hashes
        # one counter for each hash function
        self.Rs = [0] * num_hashes
        # Min-Heaps
        self.heaps = [[] for _ in range(num_hashes)]
        
        # Checking "is x in heap?"
        self.sets = [set() for _ in range(num_hashes)]

        try:
            if random_hash_functions is not None:
                self.rf = random_hash_functions
            else:
                # Hash Family Initialization if not 
                self.rf = randomhash.RandomHashFamily(count=num_hashes)
        except ImportError:
            raise ImportError("randomhash library not found. Please install it or implement a custom hash family.")

    def add(self, item):
        # hash values for all families
        hash_values = self.rf.hashes(item)
        for i in range(self.num_hashes):
            val = hash_values[i]
            
            # If val is already in the set, it's a duplicate record
            if val in self.sets[i]:
                continue
            
            # Buffer is not full!
            if len(self.heaps[i]) < self.k:
                heapq.heappush(self.heaps[i], val) # Add to heap
                self.sets[i].add(val)              # Add to set
                self.Rs[i] += 1                    # Increment counter
            
            # 'val' larger than the SMALLEST record.
            elif val > self.heaps[i][0]:
                # Remove the smallest element
                removed = heapq.heapreplace(self.heaps[i], val)
                
                # Update the set
                self.sets[i].remove(removed)
                self.sets[i].add(val)
                self.Rs[i] += 1

    def count(self):
        estimates = []
        base = 1.0 + (1.0 / self.k)
        
        for r in self.Rs:
            # If the buffer is not full == the count is exact!!
            if r < self.k:
                estimates.append(float(r))
            else:
                exponent = r - self.k
                est = self.k * (base ** exponent)
                estimates.append(int(est))
                
        return estimates
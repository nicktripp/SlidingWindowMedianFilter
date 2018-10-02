"""
    This file defines a MedianHeap data structure that tracks a median over a sliding window using two heaps.

    :author - Nick Tripp, 2018
"""
import heapq

class MedianHeap:
    """
    Like a minHeap or maxHeap, except a MedianHeap uses one of each of those to keep track of a median value.

    In particular, the maxHeap holds all elements <= the median of the heap, while the minHeap holds all elements >= the median of the heap.

    Uses lazy heap-deleting to remove elements; that is, if the element to be removed is not on top of either minHeap or maxHeap, then we simply mark it as 'dirty' and adjust an offset to note the heap that contains a new dirty element.  Then, whenever there is a new top of either heap, we check to see if it is dirty, and remove it if so.  This strategy reduces the heap delete operation from O(log(n)) to an O(1) operation, but increases worst case space complexity (and thus, worst case insertion time complexity) from O(log(m)), where m is the number of all ~current~ elements, to O(log(n)), where n is the number of all elements ~ever~ seen. However, on average, both of those remain O(log(m)).

    The number of non-dirty elements in each heap must not differ by more than 1; if so, the MedianHeap is said to be 'unbalanced', and must call self.balance().
    """

    def __init__(self):
        """
        Initializes a new empty heap.
        """
        self.max_heap = [] # All Elements <= median; top is the max of these
        self.min_heap = [] # All Elements >= median; top is the min of these
        self.offset = 0 # An integer to track the balance of how many 'extra' 'dirty' elements there are in the minHeap and maxHeap.  Specifically, this should always be equal to ((#dirty elements in minHeap) - (# dirty elements in maxHeap))
        self.dirty = {} # A dictionary of 'dirty' elements to remove if we see them later (lazy delete)

    def __str__(self):
        """
        Pretty-prints the current heap status to a string.

        NOTE: that the maxHeap elements are negated; this is because python's heapq only implements minHeaps, not maxHeaps, but we can get around that by negating every element of the maxHeap.
        """
        return "Max Heap: " + str([-x for x in self.max_heap]) + "\nMin Heap: " + str(self.min_heap) + "\nOffset: " + str(self.offset) + "\nDelete: " +  str({ k:self.dirty[k] for k in self.dirty if self.dirty[k] > 0 }) + "\nMedian: " + str(self.median())

    def max_top(self):
        """
        Peeks at the current top of the maxHeap.

        NOTE: that the result is negated; this is because python's heapq only implements minHeaps, not maxHeaps, but we can get around that by negating every element of the maxHeap.
        """
        if(self.max_empty()):
            raise RuntimeError("MedianHeap: cannot peek at empty max_heap")

        return -self.max_heap[0]

    def min_top(self):
        """ Peaks at the current top of the minHeap. """
        if(self.min_empty()):
            raise RuntimeError("MedianHeap: cannot peek at empty min_heap")

        return self.min_heap[0]

    def push_min(self, elem):
        """
        Pushes a new value to the top of the minHeap.

        Calling this method can unbalance the heap, so self.balance() should be called after this.

        :Runtime: Average O(log(m)) where m is the window size. Worst case O(log(n)) where n is the number of elements ~ever~ seen.
        """
        heapq.heappush(self.min_heap, elem)

    def push_max(self, elem):
        """
        Pushes a new value to the top of the maxHeap.

        Calling this method can unbalance the heap, so self.balance() should be called after this.

        NOTE: that the element is negated; this is because python's heapq only implements minHeaps, not maxHeaps, but we can get around that by negating every element of the maxHeap.

        :Runtime: Average O(log(m)) where m is the window size. Worst case O(log(n)) where n is the number of elements ~ever~ seen.
        """
        heapq.heappush(self.max_heap, -elem)

    def pop_min(self):
        """
        Pops the top element from the minHeap, and returns it.

        After popping, an old element is now the top of the heap; if it is dirty (a.k.a. to-be-removed), we scrub it and all subsequent tops until the top is not dirty.

        Calling this method can unbalance the heap, so self.balance() should be called after this.

        :Runtime: worst case, this operation runs in O(n) time (due to clean_top_min()), where n is the total number of elements ever seen.  However, this only happens once we build up elements to remove; over n elements this is amortized to O(1).
        """
        if(self.min_empty()):
            raise RuntimeError("MedianHeap: cannot pop from empty min_heap")

        elem = heapq.heappop(self.min_heap)
        self.clean_top_min()
        return elem

    def pop_max(self):
        """
        Pops the top element from the maxHeap, and returns it.

        After popping, an old element is now the top of the heap; if it is dirty (a.k.a. to-be-removed), we scrub it and all subsequent tops until the top is not dirty.

        Calling this method can unbalance the heap, so self.balance() should be called after this.

        NOTE: that the popped element is negated; this is because python's heapq only implements minHeaps, not maxHeaps, but we can get around that by negating every element of the maxHeap.

        :Runtime: worst case, this operation runs in O(n) time (due to clean_top_max()), where n is the total number of elements ever seen.  However, this only happens once we build up elements to remove; over n elements this is amortized to O(1).
        """
        if(self.max_empty()):
            raise RuntimeError("MedianHeap: cannot pop from empty max_heap")

        elem = -heapq.heappop(self.max_heap)
        self.clean_top_max()
        return elem

    def clean_top_max(self):
        """
        Cleans the top of the maxHeap, scrubbing each dirty element until the top of the heap is clean.

        Calling this method can unbalance the heap, so self.balance() should be called after this.

        In particular, check the top of the maxHeap to see if it is dirty. If it is, pop it, update the offset and dirty dict, and repeat.

        :Runtime: worst case, this operation runs in O(n) time, where n is the total number of elements ever seen.  However, this only happens once we build up elements to remove; over n elements this is amortized to O(1).
        """
        while (not self.max_empty() and self.is_dirty(self.max_top())):
            # Top is dirty
            self.offset += 1
            self.dirty[self.max_top()] -= 1
            heapq.heappop(self.max_heap)


    def clean_top_min(self):
        """
        Cleans the top of the minHeap, scrubbing each dirty element until the top of the heap is clean.

        In particular, check the top of the minHeap to see if it is dirty. If it is, pop it, update the offset and dirty dict, and repeat.

        Calling this method can unbalance the heap, so self.balance() should be called after this.

        :Runtime: worst case, this operation runs in O(n) time, where n is the total number of elements ever seen.  However, this only happens once we build up elements to remove; over n elements this is amortized to O(1).
        """
        while (not self.min_empty() and self.is_dirty(self.min_top())):
            # Top is dirty
            self.offset -= 1
            self.dirty[self.min_top()] -= 1
            heapq.heappop(self.min_heap)

    def is_dirty(self, elem):
        """ Returns True if the given element is marked for deletion. """
        return elem in self.dirty and self.dirty[elem] > 0

    def median(self):
        """
        Computes the median from the top of the heaps.  In particular, returns the mean of the top if each has the same # of non-'dirty' elements, otherwise, returns the top of the larger (in terms of non-'dirty' elements) heap.

        :Runtime: O(1) :)

        :return - the median of data in the heap.
        """
        if self.min_empty() and self.max_empty():
            return None

        if (len(self.min_heap) > len(self.max_heap) + self.offset):
            return self.min_top()
        elif (len(self.min_heap) < len(self.max_heap) + self.offset):
            return self.max_top()
        else:
            return (self.min_top() + self.max_top()) / 2.0

    def min_empty(self):
        """ Returns True if the minHeap has no elements. """
        return len(self.min_heap) == 0

    def max_empty(self):
        """ Returns True if the maxHeap has no elements. """
        return len(self.max_heap) == 0

    def push(self, elem):
        """
        Inserts an element into the medianHeap.

        If elem < median, the element is added to the maxHeap.
        If elem > median, the element is added to the minHeap.

        Rebalances the heap after all insertions are complete.

        :Runtime: Average O(log(m)), Worst O(log(n)), as per push_max() and push_min().
        """
        if (not self.min_empty() and elem > self.min_top()):
            # Elem in upper half, belongs in min heap
            self.push_min(elem)
        elif (not self.max_empty() and elem < self.max_top()):
            # Elem in lower half, belongs in max heap
            self.push_max(elem)
        else:
            # Elem is median, add to smaller heap
            if (len(self.min_heap) >= (len(self.max_heap) + self.offset)):
                self.push_max(elem)
            else:
                self.push_min(elem)

        self.balance()

    def is_balanced(self):
        """ Returns true if the difference between the # of non-dirty elements in each heap is <= 1. """
        return abs(len(self.max_heap) - len(self.min_heap) + self.offset) <= 1

    def balance(self):
        """
        Rebalances the heap to hold the is_balanced() invariant true. This should be called everytime after an element is added or removed from a particular heap.

        :Runtime: worst case, this operation runs in O(n) time, where n is the total number of elements ever seen.  However, this only happens once we build up elements to remove via clean_top_max/min(); over n element insertions/removals this is amortized to O(1).
        """
        while (not self.is_balanced()):
            max_size = len(self.max_heap)
            min_size = len(self.min_heap)
            if (max_size - min_size + self.offset > 1):
                # Max bigger than min, move an element from max to min
                self.push_min(self.pop_max())
            elif (max_size - min_size + self.offset < -1):
                # Min bigger than max, move an element from min to max
                self.push_max(self.pop_min())


    def remove(self, elem):
        """
        Removes a given element from the heap.

        In actuality, this function performs a 'lazy' delete; if the element to be removed is not the top of either minHeap or maxHeap, it simply marks an element as 'dirty', and expects it will be removed later (in clean_top_max/min()) when it is again encountered.

        :Runtime: O(1), but this 'lazy delete' has non-trival implications for the overall space and time performance of MedianHeap; that is, this strategy reduces the heap delete operation from O(log(n)) to an O(1) operation, but increases space complexity (and thus, insertion time complexity) from O(log(m)), where m is the number of all ~current~ elements, to O(log(n)), where n is the number of all elements ~ever~ seen.  An additional side affect is that occasionally pop_max/min() (and thus, balance()) operations run MUCH slower than normal; over many removals, this is amortized to O(1).

        Params:
        :elem - the element to remove from the heap.
            NOTE: due to the nature of lazy delete, no checks are made to see if this element is actually in the heap; if it is not, it will still be marked for future deletion, and will result in undefined behavior.
        """
        if (elem == self.min_top()):
            self.pop_min()
        elif (elem == self.max_top()):
            self.pop_max()
        else:
            if (elem in self.dirty):
                self.dirty[elem] += 1
            else:
                self.dirty[elem] = 1

            if (elem < self.median()):
                self.offset -= 1
            else:
                self.offset += 1

        self.balance()

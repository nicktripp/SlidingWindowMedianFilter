"""
    This file defines unit tests for the MedianHeap data structure found in med_heap.py.

    :author - Nick Tripp, 2018
"""

import numpy as np
import pytest

from med_heap import MedianHeap


class TestMedianHeap:
    """ Correctness Tests for MedianHeap, a rolling-median heap data structure using a minHeap and maxHeap. """

    def test_invalid_pop_peek(self):
        """ Tests popping an element from both queues when they are empty. """
        med_heap = MedianHeap()

        with pytest.raises(RuntimeError):
            med_heap.pop_min()

        with pytest.raises(RuntimeError):
            med_heap.pop_max()

        with pytest.raises(RuntimeError):
            med_heap.min_top()

        with pytest.raises(RuntimeError):
            med_heap.max_top()

    def test_balance_empty(self):
        """ Tests balancing an empty heap. """
        med_heap = MedianHeap()

        assert med_heap.is_balanced()
        med_heap.balance()
        assert med_heap.is_balanced()

    def test_median_empty(self):
        """ Tests computing the median of an empty heap. """
        med_heap = MedianHeap()

        assert med_heap.median() is None

    def test_insert_median(self):
        """ Tests insert and median functions for an extremely simple data stream. """
        med_heap = MedianHeap()

        test_data = [1,2,3,4,5]
        data_so_far = np.array([])

        for datum in test_data:
            med_heap.push(datum)
            data_so_far = np.append(data_so_far, datum)

            assert med_heap.median() == np.median(data_so_far)

    def test_insert_remove_same(self):
        """ Tests repeated insertions and removals of many identical elements. """
        med_heap = MedianHeap()

        for i in range(15):
            med_heap.push(0)
            assert med_heap.median() == 0

        for i in range(6):
            med_heap.remove(0)
            assert med_heap.median() == 0

        med_heap.push(1)
        assert med_heap.median() == 0

        for i in range(10):
            med_heap.push(1)
        assert med_heap.median() == 1

        for i in range(9):
            med_heap.remove(0)
            assert med_heap.median() == 1


    def test_remove(self):
        """
        Tests insertion into a MedianHeap, and removal in a different order than before.

        Includes an O(n) critical delete, where we remove O(n) elements at once due to the 'lazy delete' policy of MedianHeap.
        """
        med_heap = MedianHeap()

        data_so_far = np.array([])
        for i in range(1,8):
            med_heap.push(i)
            data_so_far = np.append(data_so_far, i)
            assert med_heap.median() == np.median(data_so_far)

        assert med_heap.median() == 4

        med_heap.remove(1)

        assert med_heap.median() == 4.5

        med_heap.remove(3)

        assert med_heap.median() == 5

        med_heap.remove(6)

        assert med_heap.median() == 4.5

        med_heap.remove(2)

        assert med_heap.median() == 5

        # Critical delete
        med_heap.remove(4)

        assert med_heap.median() == 6

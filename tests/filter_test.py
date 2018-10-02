"""
    This file defines unit tests for filters defined in filter.py.

    :author - Nick Tripp, 2018
"""

import pytest
import numpy as np
import random
import timeit

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from filter import RangeFilter, TemporalMedianFilter

class TestTemporalMedianFilter:
    """ Correctness Tests for TemporalMedianFilter, a sliding-window-median filter. """

    def test_init_invalid(self):
        """ Tests filter initalization with an invalid parameters. """
        # Invalid window size
        with pytest.raises(ValueError):
            filter = TemporalMedianFilter(-10,1)

        # Invalid scan_size
        with pytest.raises(ValueError):
            filter = TemporalMedianFilter(1,0)

        # Invalid filter type
        with pytest.raises(ValueError):
            filter = TemporalMedianFilter(1,1,type="NotAType")

    def test_invalid_scan_size(self):
        """ Tests calling update on a scan that is a different size than scan_size. """
        SCAN_SIZE = 10
        SCAN = range(60)

        filter = TemporalMedianFilter(3,SCAN_SIZE)


        assert SCAN_SIZE != len(SCAN)
        with pytest.raises(ValueError):
            filter.update(SCAN)

    def test_heap_update(self):
        """
        Tests basic filter updates with known data and expected medians.

        This test uses a MedianHeap to find the median of a datastream in O(log(n)) time, where n is the number of scans in the datastream.
        """
        scans = [
            np.array([0,1,2,1,3]),
            np.array([1,5,7,1,3]),
            np.array([2,3,4,1,0]),
            np.array([3,3,3,1,3]),
            np.array([10,2,4,0,0])
        ]

        expected_median = [
            np.array([0,1,2,1,3]),
            np.array([0.5,3,4.5,1,3]),
            np.array([1,3,4,1,3]),
            np.array([1.5,3,3.5,1,3]),
            np.array([2.5,3,4,1,1.5])
        ]


        filter = TemporalMedianFilter(3, 5, type=TemporalMedianFilter.TYPE_HEAP)

        for (scan, expected_median) in zip(scans, expected_median):
            np.testing.assert_array_almost_equal(filter.update(scan), expected_median)

    def test_numpy_update(self):
        """
        Tests basic filter updates with known data and expected medians.

        This test uses a numpy.median to find the median of a datastream in O(m) time, where m is the pre-defined window size.
        """
        scans = [
            np.array([0,1,2,1,3]),
            np.array([1,5,7,1,3]),
            np.array([2,3,4,1,0]),
            np.array([3,3,3,1,3]),
            np.array([10,2,4,0,0])
        ]

        expected_median = [
            np.array([0,1,2,1,3]),
            np.array([0.5,3,4.5,1,3]),
            np.array([1,3,4,1,3]),
            np.array([1.5,3,3.5,1,3]),
            np.array([2.5,3,4,1,1.5])
        ]


        filter = TemporalMedianFilter(3, 5, type=TemporalMedianFilter.TYPE_NUMPY)

        for (scan, expected_median) in zip(scans, expected_median):
            np.testing.assert_array_almost_equal(filter.update(scan), expected_median)



class TestRangeFilter:
    """ Correctness tests for RangeFilter, a min-max cropping filter. """

    def test_init_no_min_or_max(self):
        """ Tests creation of a RangeFilter without a specified maximum or minimum. """
        with pytest.raises(ValueError):
            filter = RangeFilter()

    def test_init_larger_min(self):
        """ Tests creation of a RangeFilter with a larger minimum than maximum. """
        with pytest.raises(ValueError):
            filter = RangeFilter(min=10,max=9.9999)


    def test_max(self):
        """ Tests simple input/output pairs using max"""
        MAX = 35.80
        simple_scan         = np.array([8.50, 31.48, 38.83, 12.58, 44.69, 18.79, 23.55, 29.054, 48.71, 39.947])
        expected_filtered   = np.array([8.50, 31.48, MAX,   12.58, MAX,   18.79, 23.55, 29.054, MAX,   MAX])

        max_filter      = RangeFilter(max=MAX)

        np.testing.assert_array_almost_equal(max_filter.update(simple_scan), expected_filtered)

    def test_min(self):
        """ Tests simple input/output pairs using min"""
        MIN = 20
        simple_scan         = np.array([8.50, 31.48, 38.83, 12.58, 44.69, 18.79, 23.55, 29.054, 48.71, 39.947])
        expected_filtered   = np.array([MIN,  31.48, 38.83, MIN,   44.69, MIN,   23.55, 29.054, 48.71, 39.947])

        min_filter      = RangeFilter(min=MIN)

        np.testing.assert_array_almost_equal(min_filter.update(simple_scan), expected_filtered)

    def test_max_min(self):
        """ Tests simple input/output pairs using max and min """
        MAX = 35.8
        MIN = 20
        simple_scan         = np.array([8.50, 31.48, 38.83, 12.58, 44.69, 18.79, 23.55, 29.054, 48.71, 39.947])
        expected_filtered   = np.array([MIN,  31.48, MAX,   MIN,   MAX,   MIN,   23.55, 29.054, MAX,   MAX])

        max_min_filter  = RangeFilter(max=35.8, min=20)

        np.testing.assert_array_almost_equal(max_min_filter.update(simple_scan), expected_filtered)

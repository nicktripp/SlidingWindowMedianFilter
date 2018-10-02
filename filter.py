"""
    This file defines filters to reduce noise in datastreams from LIDAR scans.

    :author - Nick Tripp, 2018
"""
import numpy as np

from med_heap import MedianHeap

class RangeFilter:
    """
    A min-max filter for streams of data

    The filter takes input data through intermittent discrete measurement scans of length 'scan_size', where 'scan_size' is within a range of ~[200,1000].

    Measured distances are between [0.03, 50].

    The update function returns an array with each entry cropped by a specified min or max.
    """
    def __init__(self, min=None, max=None):
        """
        Creates a new RangeFilter with the given specs.

        min or max can be None, but not both.

        Params:
        :min = a minimum value, below which data will be cropped up to.
        :max = a maximum value, above which data will be cropped down to.
        """
        if (min and max and min > max):
            raise ValueError("RangeFilter: min must be lower than max")

        if (not max and not min):
            raise ValueError("RangeFilter: must set either max or min")
        self.min = min
        self.max = max

    def update(self, scan):
        """
        A range filter. Numpy does all the heavy lifting.

        Params:
        :scan - an input array

        :return - a range-filtered array
        """
        return np.clip(scan, self.min, self.max)


class TemporalMedianFilter:
    """
    A sliding-window-median filter for streams of data.

    The filter takes input data through intermittent discrete measurement scans of length 'scan_size', where 'scan_size' is within a range of ~[200,1000].

    Measured distances are between [0.03, 50].

    The filter stores the data from a 'window'-sized number of scans, adding and removing to the data via the update() function.

    The update function returns an array with each entry a median of the elements at the same index of previous scans within the window.

    This median filter is implemented in two different versions (via a Median Heap or via numpy) and the type is specified in the constructor.
    """

    TYPE_HEAP  = "TYPE_HEAP"
    TYPE_NUMPY = "TYPE_NUMPY"
    TYPES = {TYPE_HEAP, TYPE_NUMPY}

    def __init__(self, window, scan_size, type=TYPE_HEAP):
        """
        Creates a new Median Filter with the given specs.

        Params:
        :window - the filter's window size. After 'window' number of calls to the update function,
        :scan_size - the fixed width of each scan of the input stream
        :type - either 'TYPE_HEAP' or 'TYPE_NUMPY', indicating this filter uses a median heap or numpy.median, respectively.
        """
        self.scans = None

        if (window < 1):
            raise ValueError("TemporalMedianFilter: window size must be > 0")
        self.window = window

        if (scan_size < 1):
            raise ValueError("TemporalMedianFilter: scan_size must be > 0")
        self.scan_size = scan_size

        if type not in TemporalMedianFilter.TYPES:
            raise ValueError("TemporalMedianFilter: type must be valid type")
        self.type= type

        self.med_heaps = [MedianHeap() for i in range(scan_size)]


    def numpy_update(self, scan):
        """
        A sliding-window-median filter using numpy.median to compute a running median.

        Params:
        :scan - an input array of size self.scan_size.

        :return - the current running-window median, computed using numpy.median over a 2D array of scans*window_size.
        """
        if (self.scans is None):
            self.scans = np.reshape(scan, (1, scan.shape[0]))
        elif (self.scans.shape[0] <= self.window):
            self.scans = np.vstack((self.scans, scan))
        else:
            self.scans = np.vstack((self.scans[1:], scan))
        return np.median(self.scans, axis=0)

    def heap_update(self, scan):
        """
        A sliding-window-median filter using MedianHeaps to compute a running median.

        Params:
        :scan - an input array of size self.scan_size.

        :return - the current running-window median, computed using a list of MedianHeap objects.
        """
        result = np.empty((self.scan_size,))

        if (self.scans is not None and self.scans.shape[0] > self.window):
            expired = self.scans[0]
        else:
            expired = None

        for idx,val in enumerate(scan):
            med_heap = self.med_heaps[idx]
            if expired is not None:
                med_heap.remove(expired[idx])
            med_heap.push(val)
            result[idx] = med_heap.median()

        if (self.scans is None):
            self.scans = np.reshape(scan, (1, scan.shape[0]))
        elif (self.scans.shape[0] <= self.window):
            self.scans = np.vstack((self.scans, scan))
        else:
            self.scans = np.vstack((self.scans[1:], scan))

        return result

    def update(self, scan):
        """ Chooses the appropriate update() method, based on the filter type. """
        if (len(scan) != self.scan_size):
            raise ValueError("TemporalMedianFilter.update(): input scan must be of size self.scan_size")

        if self.type == self.TYPE_NUMPY:
            return self.numpy_update(scan)
        elif self.type == self.TYPE_HEAP:
            return self.heap_update(scan)
        else:
            raise RuntimeError("TemporalMedianFilter: type is invalid")

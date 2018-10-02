


# Notes From Nick

**The following are notes about the implementation of this code challenge.**


---
## How To Run This Project

### Usage

You can use the MedianHeap data structure and the RangeFilter and TemporalMedianFilter objects in your own projects by importing them from their respective files;  you can also run the unit tests and benchmark tests I wrote for them.

#### Unit Tests
To run the unit tests simply run

    pytest

in this directory.  

#### Benchmark Tests
To run the benchmark tests as an analysis of the speed of various parts of this code, simply run

    python tests/benchmarks.py

**NOTE**: Running all tests as they are currently written takes a long time; on my machine, about ~30 minutes.   

If you don't want to run all tests, you can disable individual tests by commenting out the @test decorator tag prepending each test method. You can also change the test inputs to run in fewer iterations, but some implementations (i.e. MedianHeap) only show superior performance with a large number of iterations.

---
## On This Implementation

The code I wrote is in the following directory structure:

    tests/
        benchmarks.py
        filter_test.py
        heap_test.py
    filter.py
    med_heap.py
    README.md

For clarity, I'll go through each.

### ***/***

#### *filter.py*

This file defines filters to reduce noise in data streams from LIDAR scans. It includes two filters, RangeFilter, and TemporalMedianFilter.

Of the later, I implemented two types of MedianFilter:

1. one that simply uses `numpy.median`, and runs a `filter.update()` in `O(m)` time, where m is the window size.

2. one that uses a MedianHeap (defined in `med_heap.py`) to run `filter.update()` in `O(log(n))` time.  

Each implementation has pros and cons, as described below:

***numpy.median:***

Runs in asymptotic `O(M*N)` time, where `M` is the size of the window and `N` is the number of scans in the data stream.

*Pros:*
- Faster if the window size is very small.

- Space complexity is always `O(M)` where `M` is the window size.

- Time complexity is consistently `O(M)` where `M` is the window size.

- Easy to implement / debug

*Cons:*
- Slow for large window sizes and long data streams.


***MedianHeap:***

Runs in asymptotic `O(N*log(M))` time, where `N` is the number of scans in the data stream, and `M` is the window size.

*Pros:*
- Much faster for very large window sizes and very long data streams.

*Cons:*
- Space complexity is at worst `O(N)`, where `N` is the number of entries in the data stream, which can be far greater than `M`, the window size.

- Time complexity is inconsistent: at worst `update()` is `O(N)` where `N` is the number of entries in the data stream.  However, this is amortized to `O(log(N))` over `N` calls to `update()`.

***Which to Use?***

The following is an analysis of benchmark tests found in `tests/benchmark.py`:
- As the width of each scan grows (`scan_size`) increases, the comparative performance of `numpy.median` improves.  This is most likely due to more frequent and adjacent cache hits through the numpy array structure, and due to the increased constant overhead of more `MedianHeap` data structures

- As window size grows, the comparative performance of `MedianHeap` improves.

- As the number of scans in the data stream grows (`scan_count`), the comparative performance of `MedianHeap` grows.

Therefore, when `window_size << scan_count`, use `numpy.median`.  When `window_size` or `scan_count` are very large, use `MedianHeap`.  If you must guarantee a consistent running-time, use `numpy.median`.

 ---

#### *med_heap.py*

This file defines a MedianHeap data structure that tracks a median over a sliding window using two inner heaps. See this file for implementation details.

---
#### *README.md*

This is what you are reading, silly.

---
### ***/tests***

#### *benchmarks.py*

This file defines timed benchmark tests for different parts of this project.

See above for instructions on how to run.

---
#### *filter_test.py*

This file defines unit tests for filters defined in `filter.py`.

See above for instructions on how to run.

---
#### *heap_test.py*

This file defines unit tests for the MedianHeap data structure found in `med_heap.py`.

See above for instructions on how to run.

---

## Thank You!

This project proved to be a challenge, and I gained much from it.
I thank you for reading this, and for this opportunity to display my skillset!

#### Candidate Information:

    Name: Nick Tripp
    Phone: 908-343-7885
    Email: nick.e.tripp@gmail.com

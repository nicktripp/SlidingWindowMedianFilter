"""
    This file defines timed benchmark tests for different parts of this project.

    Run with 'python benchmark.py'.

    *WARNING*: This script could take a long time! On my machine, running all tests with these parameters takes ~20 minutes!  If you don't want to run all tests, you can disable individual tests by commenting out the @test decorator tag prepending each test method. You can also change the test inputs to run in fewer iterations, but some implementations (i.e. MedianHeap) only show superior performance with a large number of iterations.


    Benchmarked modules include:
        - MedianHeap
        - TemporalMedianFilter

    :author - Nick Tripp, 2018
"""
import timeit, time
import os, sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from med_heap import MedianHeap
from filter import RangeFilter, TemporalMedianFilter

##############################################################
##################   BENCHMARK SETUP   #################$$####
##############################################################

###
# Benchmark globals
###
suites = {} # Maps suite names to lists of test suite functions
suite_descriptions = {}
# Test Suite Names:
MEDIAN_HEAP = "MedianHeap"
TEMPORAL_FILTER = "TemporalMedianFilter"


###
# DECORATORS
###

def test(name):
    """
    A decorator for bench tests.

    Adds the marked test to the test suite holding the specified name.

    Each marked test must take no parameters.

    Each marked test should run its benchmark, printing to standard output, and return nothing.
    """

    def add_test_to_suite(func):
        if name not in suites:
            suites[name] = [func]
        else:
            suites[name].append(func)

    return add_test_to_suite

###
# Helper functions
###

class color:
    """ A class for constants to stylize terminal output. """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header():
    """ Prints a header for this benchmark testing session. """
    rows, columns = os.popen('stty size', 'r').read().split()
    head_str=color.BOLD + "{:=^" + columns + "}" + color.END
    print(head_str.format(" Running Benchmark Test Suites "))

def print_footer(tests_run, time_eleapsed):
    """
    Prints a footer for this benchmark testing session.

    Includes the total number of tests run and total time elapsed.

    Params
    :tests_run - the total number of ran tests to print.
    :time_elapsed - the total amount of elapsed time, in seconds, to print.
    """
    rows, columns = os.popen('stty size', 'r').read().split()
    head_str=color.BOLD + color.GREEN + "{:=^" + columns + "}" + color.END
    print(head_str.format(" {} Tests Run, {} seconds elapsed ".format(tests_run, time_eleapsed)))

def run_suite(suite_name):
    """
    Prints a header for the given test suite, then runs all tests under that suite name.

    Params:
    :suite_name - the name of the test suite to run.  Any given name string must match that passed to a @test decorator.

    :return - the number of tests ran in this suite.
    """
    if (suite_name not in suites):
        raise ValueError("Suite '{}' does not exist".format(suite_name))

    rows, columns = os.popen('stty size', 'r').read().split()
    head_str=color.PURPLE + "{:=^" + columns + "}" + color.END
    print(head_str.format(" Bench-Test Suite: {} ".format(suite_name)))
    if (suite_name in suite_descriptions):
        print(suite_descriptions[suite_name])

    TESTS = suites[suite_name]

    for test in TESTS:
        print_bench_test(test)

    tests_count = len(TESTS)
    head_str= color.PURPLE + "{:-^" + columns + "}" + color.END
    print(head_str.format(" {}: {} Tests Run ".format(suite_name, tests_count)))
    return tests_count

###
# WRAPPER FUNCTIONS
###
def print_bench_test(func):
    """
    Prints a header for the given test, then runs it.

    Sepcifically, the header consists of the name of the test as specified by the name of the passed function and the doctstring of the given function.

    Params
    :func - a function defining the benchmark test to run. Must be a named function.
    """
    rows, columns = os.popen('stty size', 'r').read().split()

    head_str="{:_^" + columns + "}"
    print(head_str.format(" {} ".format(func.__name__)))
    if (func.__doc__ is not None):
        print(func.__doc__)
        head_str="{:.^" + columns + "}"
    print(head_str.format("") + "\n")
    func()
    head_str="{:.^" + columns + "}"
    print("\n" + head_str.format("") + "\n")


##############################################################
#####################   TEST SUITES   ########################
##############################################################

##########################
# MEDIAN HEAP TEST SUITE #
##########################
@test(MEDIAN_HEAP)
def test_push_speed():
    """
    Tests the speed of different sets of many insertions to MedianHeap compared to the same insertions over a python heapq heap.

    The insertions are randomized on each run of the test, but identically seeded across insertion sets within the same test.

    Insertion on both MedianHeap and heapq should run in O(log(n)) time; inserting n numbers should yield a running time of O(nlog(n)).

    EXPECTED BEHAVIOUR: insertion times for Median Heap and heapq should be comparable.
    """
    ### COMMON SETUP ###
    SEED = random.randrange(sys.maxsize)
    ITERATIONS = [ 10**i for i in range(3,7) ] # List of numbers of iterations to run

    ### MEDIAN HEAP ###
    setup= """
from med_heap import MedianHeap
import random
med_heap = MedianHeap()
    """
    stmt = """
random.seed({})
med_heap.push(random.uniform(0.03,50))
        """.format(SEED)

    med_heap_times = { i:timeit.timeit(setup=setup,stmt=stmt,number=i) for i in ITERATIONS }

    ### HEAPQ ###
    setup="""
import heapq, random
heap = []
        """
    stmt = """
random.seed({})
elem = random.uniform(0.03,50)
heapq.heappush(heap,elem)
heapq.heappop(heap)
heapq.heappush(heap, elem)
        """ .format(SEED)

    heapq_times = { i:timeit.timeit(setup=setup,stmt=stmt,number=i) for i in ITERATIONS }


    ### PRINT RESULTS ###
    print(color.UNDERLINE + "{:<15}{:<15}{:<15}".format("# Of Inserts","MedHeap","Heapq") + color.END)
    for i in ITERATIONS:
        print("{:<15}{:<15.7f}{:<15.7f}".format(i,med_heap_times[i],heapq_times[i]))



#####################################
# TEMPORAL MEDIAN FILTER TEST SUITE #
#####################################

###
# HELPER FUNCS
###

def time_filter_with_params(window,scan_count,scan_size,seed):
    """
    Helper function to run a benchmark with given filering parameters.

    Filters a random data stream of width 'scan_size', with 'scan_count' entries using a filter with a window size 'window'.
    The data stream is seeded with the given 'seed'.

    The total time to filter all 'scan_count' scans using each of the types of TemporalMedianFilter is recorded and returned.

    Params:
    :window - the size of the window to pass to the filter
    :scan_size - the width of each scan
    :scan_count - the total final number of scans in the data stream
    :seed - the seed for our psuedo-random number generator.

    :return (int,int) - a tuple of the time it takes to filter the data stream using MedianHeap and numpy.median, respectively.
    """

    ### COMMON SETUP ###
    SCAN_COUNT = scan_count
    params = {
        "SCAN_SIZE": scan_size, # Width of each scan
        "WINDOW": window, # window size
        "SEED": seed,
        "TYPE": TemporalMedianFilter.TYPE_HEAP
    }
    base_setup="""
import numpy as np
import random
from filter import TemporalMedianFilter

filter = TemporalMedianFilter(window={WINDOW}, scan_size={SCAN_SIZE}, f_type="{TYPE}")

random.seed({SEED})
        """
    stmt = """
scan = np.array([random.uniform(0.03,50) for x in range({SCAN_SIZE})])
filter.update(scan)
        """.format(**params)

    ### MEDIAN HEAP TIME ###
    setup=base_setup.format(**params)

    med_heap_time = timeit.timeit(stmt=stmt,setup=setup, number=SCAN_COUNT)

    ### NUMPY MEDIAN TIME ###
    params["TYPE"] = TemporalMedianFilter.TYPE_NUMPY
    setup=base_setup.format(**params)

    numpy_time = timeit.timeit(stmt=stmt, setup=setup, number=SCAN_COUNT)

    return med_heap_time, numpy_time

def test_iterations_with_params(scan_counts, scan_size, window_ratio, window=None):
    """
    Helper function that runs Median Filter updates over each member of scan_counts, and pretty-prints the result.

    Adjusting scan_counts and window_ratio affects which type of TemporalMedianFilter performs better - in each case, a large scan_count or window_ratio results in the MedianHeap filter outperforming the numpy.median filter.

    Params:
    :scan_counts list(ints) - an array of scan sets to run.  Each element represents another set of 'element' # of scans to run.
    :scan_size - the width of each scans to run.
    :window_ratio - the ratio of the window to the scan_count
    :window - if a specific window is specified, window_ratio is ignored, and window is used instead.
    """
    SEED = random.randrange(sys.maxsize)

    exec_times = { i:(time_filter_with_params(window=(window if window is not None else i*window_ratio), scan_count=i, scan_size=scan_size, seed=SEED)) for i in scan_counts}

    ### PRINT RESULTS ###
    print("Test Parameters")
    print(":Scan Width: {} elements".format(scan_size))
    if window is not None:
        print(":Window Size: {}\n".format(window))
    else:
        print(":Window Size: {}% of scan count\n".format(window_ratio*100))
    print(color.UNDERLINE + "{:<15}{:<15}{:<15}{:<15}".format("Scan Count","MedHeap (s)","Numpy (s)","Faster Version") + color.END)

    for i in scan_counts:
        print("{:<15}{:<15.7f}{:<15.7f}{:<15}".format(i,exec_times[i][0],exec_times[i][1],color.GREEN + "Median Heap" + color.END if exec_times[i][0] < exec_times[i][1] else color.RED + "Numpy.median" + color.END ))



###
# BENCH TESTS
###
suite_descriptions[TEMPORAL_FILTER] =     """
    Tests different versions of TemporalMedianFilter.update(), including numpy.median and MedianHeap filtering.

    Insertions via update() are randomized on each run of the test, but identically seeded within a test.

    Runtime comparisions:
    (where 'm' is the window size and 'n' is the total number of scans)
    MedianHeap (average):
        - per update(): O(log(m))
        - total:        O(n*log(m))
    numpy.median:
        - per update(): O(m)
        - total:        O(n*m)
    """



@test(TEMPORAL_FILTER)
def test_1DataStream_10percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,6)]
    SCAN_SIZE = 1
    WINDOW_RATIO = 0.1

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_1DataStream_50percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """

    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,6)]
    SCAN_SIZE = 1
    WINDOW_RATIO = 0.5

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_1DataStream_100percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,6)]
    SCAN_SIZE = 1
    WINDOW_RATIO = 1.0

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_10DataStream_10percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,6)]
    SCAN_SIZE = 10
    WINDOW_RATIO = 0.1

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_10DataStream_50percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,5)]
    SCAN_SIZE = 10
    WINDOW_RATIO = 0.5

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_10DataStream_100percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,5)]
    SCAN_SIZE = 10
    WINDOW_RATIO = 1.0

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)


@test(TEMPORAL_FILTER)
def test_200DataStream_10percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,5)]
    SCAN_SIZE = 200
    WINDOW_RATIO = 0.1

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_200DataStream_50percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,5)]
    SCAN_SIZE = 200
    WINDOW_RATIO = 0.5

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_200DataStream_100percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,5)]
    SCAN_SIZE = 200
    WINDOW_RATIO = 1.0

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_1000DataStream_10percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,5)]
    SCAN_SIZE = 1000
    WINDOW_RATIO = 0.1

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_1000DataStream_50percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,4)] + [2000]
    SCAN_SIZE = 1000
    WINDOW_RATIO = 0.5

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_1000DataStream_100percentWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap perfoms increasingly better than Numpy.median as scan_count and/or window_size grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,4)] + [2000]
    SCAN_SIZE = 1000
    WINDOW_RATIO = 1.0

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, WINDOW_RATIO)

@test(TEMPORAL_FILTER)
def test_1000DataStream_fixedSmallWindow():
    """
    EXPECTED BEHAVIOR: Numpy.median runs much faster than MedianHeap for this small window_size.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,4)]
    SCAN_SIZE = 1000
    WINDOW = 10

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, None, window=WINDOW)

@test(TEMPORAL_FILTER)
def test_1000DataStream_fixedLargeWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap runs increasingly faster than numpy.median as scan_count grows larger.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,5)]
    SCAN_SIZE = 1000
    WINDOW = 1000

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, None, window=WINDOW)

@test(TEMPORAL_FILTER)
def test_10DataStream_fixedSmallWindow():
    """
    EXPECTED BEHAVIOR: MedianHeap is slower than numpy.median for this small window size.
    """
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,6)]
    SCAN_SIZE = 10
    WINDOW = 10

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, None, window=WINDOW)

@test(TEMPORAL_FILTER)
def test_10DataStream_fixedLargeWindow():
    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,6)]
    SCAN_SIZE = 10
    WINDOW = 1000

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, None, window=WINDOW)

@test(TEMPORAL_FILTER)
def test_1DataStream_fixedLargeWindow():

    ### SETUP ###
    ITERATIONS = [10**x for x in range(1,6)]
    SCAN_SIZE = 1
    WINDOW = 1000

    test_iterations_with_params(ITERATIONS, SCAN_SIZE, None, window=WINDOW)


#####################
#######  MAIN  ######
#####################


def main():
    tests_run = 0

    print_header()

    start = time.time()

    for suite in suites:
        tests_run += run_suite(suite)

    end = time.time()
    elapsed = end - start

    print_footer(tests_run, elapsed)

if __name__ == '__main__':
    main()

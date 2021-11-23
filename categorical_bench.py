import cudf
import cupy as cp

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
import rmm

import pandas as pd

import time
import itertools
import argparse

def rmm_pool():
    nvmlInit()
    free_mem = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).free
    pool_size = 0.95 * free_mem
    pool_size -= pool_size%256
    cudf.set_allocator(pool=True, initial_pool_size=pool_size)
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

def warm_up():
    _ = cp.array([1, 2, 3])
    _ = cudf.Series([1, 2, 3])

def set_category_worst_bench(cardinality, n_rows):
    """
    Benchmarks `cat.set_category` efficiency, in worst case

    Example in small scale:
    old_cats: [0, 1, 2, 3, 4]
    old_codes: [0, 1, 2, 3, 4]

    new_cats: [2.0, 3.0, 4.0, 5.0, 6.0]
    new_codes: [-1, -1, 0, 1, 2]
    """
    old_categories = cp.arange(0, cardinality).astype("int")
    old_items = cp.tile(old_categories,  int(n_rows / cardinality))
    old = cudf.Series(old_items, dtype="category")

    new_categories = cp.arange(int(cardinality / 2), int(cardinality / 2)+cardinality).astype("float32")

    t = time.time()
    new = old.cat.set_categories(new_categories)
    t = time.time() - t

    return t

def set_category_avg_bench(cardinality, n_rows):
    """
    Benchmarks `cat.set_category` efficiency, in avg case

    Example in small scale:
    old_cats: [0, 1, 2, 3, 4]
    old_codes: [0, 1, 2, 3, 4]

    new_cats: [2, 3, 4, 5, 6]
    new_codes: [0, 1, 2, -1, -1]
    """
    old_categories = cp.arange(0, cardinality)
    old_items = cp.tile(old_categories,  int(n_rows / cardinality))
    old = cudf.Series(old_items, dtype="category")

    new_categories = cp.arange(int(cardinality / 2), int(cardinality / 2)+cardinality)

    t = time.time()
    new = old.cat.set_categories(new_categories)
    t = time.time() - t

    return t

def join_bench(cardinality, n_rows):
    """
    Joins two tables using categorical column as key
    """

    categories = cp.arange(0, cardinality)
    items = cp.tile(categories,  int(n_rows / cardinality))
    key = cudf.Series(items, dtype="category")

    payload = cudf.Series(cp.arange(0, n_rows))
    
    lhs = cudf.DataFrame({'key': key, 'lpayload': payload})

    rkey = key[::-1]
    rhs = cudf.DataFrame({'key': rkey, 'rpayload': payload})

    t = time.time()
    _ = lhs.join(rhs, rsuffix="_r")
    t = time.time() - t

    return t

def concat_bench(cardinality, n_rows):
    """
    Concats two series, wher lhs, rhs
    """

    cat = cp.arange(0, cardinality)

    litems = cp.tile(cat,  int(n_rows / cardinality))
    lhs = cudf.Series(litems, dtype="category")

    ritems = cp.tile(cat,  int(n_rows / cardinality))
    rhs = cudf.Series(ritems, dtype="category")

    t = time.time()
    _ = cudf.concat([lhs, rhs])
    t = time.time() - t

    return t

def sort_values_bench(cardinality, n_rows):
    """
    Sort a categorical column
    """

    cat = cp.arange(0, cardinality)
    items = cp.tile(cat,  int(n_rows / cardinality))
    s = cudf.Series(items, dtype="category")

    t = time.time()
    _ = s.sort_values()
    t = time.time() - t

    return t

def fillna_bench(cardinality, n_rows):
    """
    Sort a categorical column
    """

    cat = cp.arange(0, cardinality)
    items = cp.tile(cat,  int(n_rows / cardinality))
    s = cudf.Series(items, dtype="category")

    # This makes half of `s` <NA>
    s = s.cat.set_categories(cp.arange(0, cardinality / 2))

    fill_values = cudf.Series(cp.full(n_rows, 0), dtype=s.dtype)

    t = time.time()
    _ = s.fillna(fill_values)
    t = time.time() - t

    return t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('flag', type=str, help="Before or after")
    args = parser.parse_args()

    rmm_pool()
    warm_up()

    repeat = 20
    total_t = 0

    # n_rows = [10]
    # n_categories = [10]

    n_rows = [100, 100_000, 100_000_000]
    n_categories = [100, 100_000, 100_000_000]

    results = pd.DataFrame()

    for n_row, n_cat in itertools.product(n_rows, n_categories):
        if n_row >= n_cat:
            for _ in range(repeat):
                total_t += fillna_bench(n_cat, n_row)
            avg_t = total_t / repeat
            
            results.loc[n_row, n_cat] = avg_t

    results.to_json(f"categorical_bench_{args.flag}.json")

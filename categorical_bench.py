import cudf
import cupy as cp
import pytest


# @pytest.fixture(params=[100, 100_000, 100_000_000])
# def cardinality(request):
#     return request.param

# @pytest.fixture(params=[100, 100_000, 100_000_000])
# def n_rows(request):
#     return request.param

@pytest.mark.parametrize("cardinality", [100, 100_000, 100_000_000])
@pytest.mark.parametrize("n_rows", [100, 100_000, 100_000_000])
def set_category_requires_promote_type_bench(benchmark, cardinality, n_rows):
    """
    Benchmarks `cat.set_category` efficiency

    The new numeric category types are different from old cats.
    And one of them require to be promoted to the matching type.

    Example in small scale:
    old_cats: [0, 1, 2, 3, 4]
    old_codes: [0, 1, 2, 3, 4]

    new_cats: [2.0, 3.0, 4.0, 5.0, 6.0]
    new_codes: [-1, -1, 0, 1, 2]
    """
    if cardinality > n_rows:
        pytest.skip()

    old_categories = cp.arange(0, cardinality).astype("int")
    old_items = cp.tile(old_categories,  int(n_rows / cardinality))
    old = cudf.Series(old_items, dtype="category")

    new_categories = cp.arange(int(cardinality / 2), int(cardinality / 2)+cardinality).astype("float32")

    benchmark(old.cat.set_categories, new_categories)

@pytest.mark.parametrize("cardinality", [100, 100_000, 100_000_000])
@pytest.mark.parametrize("n_rows", [100, 100_000, 100_000_000])
def set_category_no_promote_type_bench(benchmark, cardinality, n_rows):
    """
    Benchmarks `cat.set_category` efficiency

    Example in small scale:
    old_cats: [0, 1, 2, 3, 4]
    old_codes: [0, 1, 2, 3, 4]

    new_cats: [2, 3, 4, 5, 6]
    new_codes: [0, 1, 2, -1, -1]
    """
    if cardinality > n_rows:
        pytest.skip()

    old_categories = cp.arange(0, cardinality)
    old_items = cp.tile(old_categories,  int(n_rows / cardinality))
    old = cudf.Series(old_items, dtype="category")

    new_categories = cp.arange(int(cardinality / 2), int(cardinality / 2)+cardinality)

    benchmark(old.cat.set_categories, new_categories)

@pytest.mark.parametrize("cardinality", [100, 100_000, 100_000_000])
@pytest.mark.parametrize("n_rows", [100, 100_000, 100_000_000])
def join_bench(benchmark, cardinality, n_rows):
    """
    Joins two tables using categorical column as key
    """
    if cardinality > n_rows:
        pytest.skip()

    categories = cp.arange(0, cardinality)
    items = cp.tile(categories,  int(n_rows / cardinality))
    key = cudf.Series(items, dtype="category")

    payload = cudf.Series(cp.arange(0, n_rows))
    
    lhs = cudf.DataFrame({'key': key, 'lpayload': payload})

    rkey = key[::-1]
    rhs = cudf.DataFrame({'key': rkey, 'rpayload': payload})

    benchmark(lhs.join, rhs, rsuffix="_r")

@pytest.mark.parametrize("cardinality", [100, 100_000, 100_000_000])
@pytest.mark.parametrize("n_rows", [100, 100_000, 100_000_000])
def concat_bench(benchmark, cardinality, n_rows):
    """
    Concats two series, wher lhs, rhs
    """
    if cardinality > n_rows:
        pytest.skip()

    cat = cp.arange(0, cardinality)

    litems = cp.tile(cat,  int(n_rows / cardinality))
    lhs = cudf.Series(litems, dtype="category")

    ritems = cp.tile(cat,  int(n_rows / cardinality))
    rhs = cudf.Series(ritems, dtype="category")

    benchmark(cudf.concat, [lhs, rhs])

@pytest.mark.parametrize("cardinality", [100, 100_000, 100_000_000])
@pytest.mark.parametrize("n_rows", [100, 100_000, 100_000_000])
def sort_values_bench(benchmark, cardinality, n_rows):
    """
    Sort a categorical column
    """
    if cardinality > n_rows:
        pytest.skip()

    cat = cp.arange(0, cardinality)
    items = cp.tile(cat,  int(n_rows / cardinality))
    s = cudf.Series(items, dtype="category")

    benchmark(s.sort_values)

@pytest.mark.parametrize("cardinality", [100, 100_000, 100_000_000])
@pytest.mark.parametrize("n_rows", [100, 100_000, 100_000_000])
def fillna_bench(benchmark, cardinality, n_rows):
    """
    Sort a categorical column
    """
    if cardinality > n_rows:
        pytest.skip()

    cat = cp.arange(0, cardinality)
    items = cp.tile(cat,  int(n_rows / cardinality))
    s = cudf.Series(items, dtype="category")

    # This makes half of `s` <NA>
    s = s.cat.set_categories(cp.arange(0, cardinality / 2))

    fill_values = cudf.Series(cp.full(n_rows, 0), dtype=s.dtype)

    benchmark(s.fillna, fill_values)

"""Import before lightgbm: hide the broken dask of this anaconda (lightgbm then skips its dask support)."""
import sys
for m in ("dask", "dask.array", "dask.dataframe", "dask.distributed"):
    sys.modules[m] = None

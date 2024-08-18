import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from gluonts.dataset.arrow import ArrowWriter

parser = argparse.ArgumentParser("process_data")
parser.add_argument("path", help="Path to data", type=str)
args = parser.parse_args()


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and time_series.ndim == 2
    )

    # TODO: set start time
    start = np.datetime64("1970-01-01 00:00", "s")

    dataset = [{"start": start, "target": ts} for ts in time_series]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


if __name__ == "__main__":
    time_series = pd.read_csv(args.path)
    time_series_array = time_series["TotalCount"].to_list()

    # Convert to GluonTS arrow format
    convert_to_arrow("./data/training_data.arrow", time_series=time_series_array)

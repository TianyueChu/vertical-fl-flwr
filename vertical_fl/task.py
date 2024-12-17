from pathlib import Path
from logging import WARN
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn as nn
from flwr.common.logger import log

from datasets import Dataset

NUM_VERTICAL_SPLITS = 2


def load_data( data_name: str = 'bank'):
    """Read the data from bank or tele company dataset.

    """

    if NUM_VERTICAL_SPLITS != 2:
        log(
            WARN,
            "To run this example with num_partitions other than 2, you need to update how "
            "the Vertical FL training is performed. This is because the shapes of the "
            "gradients might not be the same along the first dimension.",
        )

    df = pd.read_csv(Path(__file__).parents[1] / f"data/{data_name}.csv")
    processed_df = df.drop("ID", axis=1).copy()

    # Read the data
    v_partition = processed_df

    # Covert to HuggingFace dataset
    dataset = Dataset.from_pandas(v_partition)

    # Extract partition of the `ClientApp` calling this function
    partition = dataset

    return partition.to_pandas()


class ClientModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 16)

    def forward(self, x):
        return self.fc(x)

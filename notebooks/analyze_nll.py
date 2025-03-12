# %%
from pathlib import Path

import numpy as np
import pandas as pd

file = "results/autoregressive 2025-03-11 13:06:28.014927"
file = Path(file)

data = np.load(file / "full_log.npz")

strategies = list(data.keys())

df = pd.DataFrame.from_dict(
    {strategy: np.reshape(nll, -1) for strategy, nll in data.items()}
)
df.describe()

# %%

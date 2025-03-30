# %%
from pathlib import Path

import numpy as np
import pandas as pd

# file = "results/autoregressive 2025-03-12 14:55:22.979512"  # trained on noisy (old format)
# file = "results/autoregressive 2025-03-13 11:00:52.022274"  # trained on noisy
# file = "results/autoregressive 2025-03-13 11:02:20.878417"  # trained on underlying GP
file = "results/autoregressive 2025-03-17 14:31:37.439478"  # noisy test, TNP-KR-Scan
# file = "results/autoregressive 2025-03-21 18:16:36.095384"  # consistency 2, 0.1
# file = "results/autoregressive 2025-03-21 20:40:26.697617"  # consistency 10, 0.5
file = Path(file)

data = np.load(file / "full_log.npz")

strategies = list(data.keys())

df = pd.DataFrame.from_dict(
    {strategy: np.reshape(nll, -1) for strategy, nll in data.items()}
)

# %%
df.mean().sort_values()
#        diagonal	ltr          closest	  furthest	    random
# mean	-5.085879	-55.083778	-55.513279	-56.163029	-56.967819

# Entropy is ~50, so
# so |ltr - random| is a significant difference in terms of divergence.

# %%
# df[[col for col in df.columns if col.startswith("nll_gp")]].mean()
# %%

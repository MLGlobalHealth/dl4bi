# Reproduce all TNP-KR Paper Results
`python reproduce_paper.py tnp_kr [--dry_run]`

# Gaussian Processes
`python gp.py data=1d kernel=rbf model=tnp_kr_full seed=7 [wandb=False] [+name="Experiment name"]`

# Bayesian Optimization
`python bayes_opt.py data=1d kernel=rbf model=tnp_kr_full seed=7 [wandb=False] [+name="Experiment name"]`

# Population Genetics
`python popgen.py model=tnp_kr_full seed=7 [wandb=False] [+name="Experiment name"]`

# MNIST
`python mnist.py model=tnp_kr_full seed=7 [wandb=False] [+name="Experiment name"]`

# CelebA
`python celeba.py model=tnp_kr_full seed=7 [wandb=False] [+name="Experiment name"]`

NOTE: The Tensorflow Dataset `celeb_a` is broken (invalid checksum), so this script downloads the files directly from the [source](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) on Google Drive. However, sometimes this also fails due to Google Drive limits (many people download this dataset from scripts). If this happens, you can download the images directly [here](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) and the dataset partition list [here](https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=drive_link&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg). You will need to put these in the `dl4bi/benchmarks/meta_regression/cache/celeba` directory and rerun the `celeba.py` script.

# Heaton
1. Download the `AllSatelliteTemps.RData` and `AllSimulatedTemps.Rdata` datasets [here](https://github.com/finnlindgren/heatoncomparison/tree/master/Data).
2. Make a Heaton directory: `mkdir -p ~/dl4bi/benchmarks/meta_regression/cache/heaton`
3. Convert the RData to CSVs and save to the cache directory:
```R
library(tidyverse)
load("AllSimulatedTemps.RData")
load("AllSatelliteTemps.RData")
path <- "~/dl4bi/benchmarks/meta_regression/cache/heaton/"
all.sim.data %>% write_csv(str_c(path, "sim.csv"))
all.sat.temps %>% write_csv(str_c(path, "sat.csv"))
```

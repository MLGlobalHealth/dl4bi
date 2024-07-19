# Gaussian Processes
`python gp.py +exp=1d +kernel=rbf +model=sptx_full +seed=7 [+wandb=True] [+name="Experiment name"]`

# Bayesian Optimization
`python bayes_opt.py +exp=1d +kernel=rbf +model=sptx_full +seed=7 [+wandb=True] [+name="Experiment name"]`

# MNIST
`python mnist.py +model=sptx_full +seed=7 [+wandb=True] [+name="Experiment name"]`

# CelebA
`python celeba.py +model=sptx_full +seed=7 [+wandb=True] [+name="Experiment name"]`

NOTE: The Tensorflow Dataset `celeb_a` is broken (invalid checksum), so this script downloads the files directly from the [source](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) on Google Drive. However, sometimes this also fails due to Google Drive limits (many people download this dataset from scripts). If this happens, you can download the images directly [here](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) and the dataset partition list [here](https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=drive_link&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg). You will need to put these in the `dsp/benchmarks/meta_regression/cache/celeba` directory and rerun the `celeba.py` script.

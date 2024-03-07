from dataclasses import dataclass

import flax.linen as nn


class DeepChol(nn.Module):
    model: nn.Module

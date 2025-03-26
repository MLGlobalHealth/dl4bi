# %%
from json import dumps

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from dl4bi.meta_learning.autoregressive import ARSampler
from dl4bi.meta_learning.autoregressive.analytic import (
    analytic_gp,
)
from dl4bi.meta_learning.train_utils import build_gp_dataloader, instantiate, load_ckpt

path = "results/1d/rbf/42/TNP-KR: Attention.ckpt"
path_trained_on_noisy_test = "results/1d/rbf/42/TNP-KR: Attention noisy test.ckpt"


state, config = load_ckpt(path)
print(dumps(OmegaConf.to_object(config), indent=2))

sampler_1 = ARSampler.from_state(state)
sampler_2 = ARSampler.from_state(load_ckpt(path_trained_on_noisy_test)[0])
# %%
config.data.batch_size = 1
config.data.num_ctx = {"min": 1, "max": 3}

rng = jax.random.key(42)
dataloader = build_gp_dataloader(config.data, config.kernel)(rng)

# %%%
s_ctx, f_ctx, valid_lens_ctx, s, f, _valid_lens_test, var, ls, _period = next(
    dataloader
)
obs_noise = config.data.obs_noise
kernel = instantiate(config.kernel.kwargs.kernel)

num_ctx = config.data.num_ctx.max
s_test = s
idx = jnp.argsort(s, axis=1)
s_test = jnp.take_along_axis(s_test, idx, axis=1)
L_ctx = int(valid_lens_ctx[0])


mean_1, sd_1 = sampler_1.model(s_ctx, f_ctx, s_test, valid_lens_ctx)
mean_1, sd_1 = mean_1.squeeze(), sd_1.squeeze()

mean_2, sd_2 = sampler_2.model(s_ctx, f_ctx, s_test, valid_lens_ctx)
mean_2, sd_2 = mean_2.squeeze(), sd_2.squeeze()

s_ctx = s_ctx.squeeze()[:L_ctx]
f_ctx = f_ctx.squeeze()[:L_ctx]
s_test = s_test.squeeze()

mean_gp, var_gp = analytic_gp(s_ctx, f_ctx, s_test, kernel, var, ls, obs_noise)
pointwise_var_gp = jnp.diag(var_gp)
pointwise_sd_gp = pointwise_var_gp**0.5

fig, axs = plt.subplots(1, L_ctx, sharey=True, figsize=(3 * L_ctx, 3))
for i, ax in enumerate(axs):
    ax.plot(s_test, sd_1, "C0", label="model (trained without noise in f_test)")
    ax.plot(s_test, sd_2, "C1", label="model (trained with noisy f_test)")
    ax.plot(s_test, pointwise_sd_gp, "C0--", alpha=0.5, label="analytic GP")
    ax.plot(
        s_test,
        (pointwise_var_gp + obs_noise**2) ** 0.5,
        "C1--",
        alpha=0.5,
        label="analytic OBS",
    )
    ax.set_xlim(s_ctx[i] - 0.05, s_ctx[i] + 0.05)
    ax.set_title(f"s_ctx[i] = {s_ctx[i]:.2f}")
    ax.set_ylim(0, 0.4)

axs[0].set_ylabel("predicitve SD")

fig.suptitle("Point-wise standard deviations around context points")
fig.legend(
    *axs[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1, 0)
)

fig.set_constrained_layout(True)
plt.show()

# %%
plt.title("Mean")
plt.plot(s_test, mean_1)
plt.plot(s_test, mean_2)
plt.plot(s_ctx, f_ctx, "+")
plt.show()
# %%
plt.title("SD")
plt.plot(s_test, sd_1)
plt.plot(s_test, sd_2)
plt.plot(s_test, pointwise_sd_gp, "C0--")
plt.plot(s_test, (pointwise_var_gp + obs_noise**2) ** 0.5, "C1--")
plt.show()
# %%

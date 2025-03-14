import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state


# ----------------------------
# 1. Transformer Encoder Module
# ----------------------------
class TransformerEncoder(nn.Module):
    model_dim: int
    num_layers: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len, feature_dim)
        # First project the input to model_dim.
        x = nn.Dense(self.model_dim)(x)
        # (For simplicity, we skip positional encodings.)
        for _ in range(self.num_layers):
            # A simple self-attention layer.
            x = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.model_dim)(
                x
            )
            # A simple feed-forward layer with a residual connection.
            x = x + nn.Dense(self.model_dim)(x)
        # For conditioning, we take the representation of the first token.
        context = x[:, 0, :]  # (batch, model_dim)
        return context


# ----------------------------
# 2. Spline Conditioner Module
# ----------------------------
class SplineConditioner(nn.Module):
    num_bins: int
    hidden_dim: int

    @nn.compact
    def __call__(self, context):
        # context: (batch, context_dim)
        x = nn.Dense(self.hidden_dim)(context)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        # Output unnormalized parameters:
        # For each bin: widths and heights; for derivatives, we output (num_bins+1) values.
        widths = nn.Dense(self.num_bins)(x)
        heights = nn.Dense(self.num_bins)(x)
        derivatives = nn.Dense(self.num_bins + 1)(x)
        return widths, heights, derivatives


# ----------------------------
# 3. Simplified Rational Quadratic Spline Transform
# ----------------------------
def rational_quadratic_spline(
    x,
    widths,
    heights,
    derivatives,
    left,
    right,
    bottom,
    top,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    inverse=False,
):
    """
    A simplified (almost linear) version of a rational quadratic spline.
    x: (batch,) values assumed in [left, right]
    widths, heights: (batch, num_bins)
    derivatives: (batch, num_bins+1)
    Returns: y and log_det (each shape (batch,))
    Note: A full implementation would use nonlinear (rational quadratic) formulas.
    """
    num_bins = widths.shape[-1]
    # Normalize widths and heights so that they sum to (right-left) and (top-bottom)
    widths = jax.nn.softmax(widths, axis=-1)
    heights = jax.nn.softmax(heights, axis=-1)
    widths = widths * ((right - left) - num_bins * min_bin_width) + min_bin_width
    heights = heights * ((top - bottom) - num_bins * min_bin_height) + min_bin_height
    # Ensure derivatives are positive.
    derivatives = jax.nn.softplus(derivatives) + min_derivative

    # Compute bin edges (for x and y)
    # x_bins: (batch, num_bins+1)
    x_bins = jnp.concatenate(
        [jnp.full((x.shape[0], 1), left), jnp.cumsum(widths, axis=-1) + left], axis=-1
    )
    y_bins = jnp.concatenate(
        [jnp.full((x.shape[0], 1), bottom), jnp.cumsum(heights, axis=-1) + bottom],
        axis=-1,
    )

    # For each x, find the bin it belongs to.
    # For each batch element, count how many bin edges are <= x.
    bin_idx = jnp.sum(x[..., None] >= x_bins, axis=-1) - 1  # (batch,)
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)

    # Define a helper to gather parameters per batch element.
    def gather_param(param, param_bins):
        # param: shape (batch, num_bins) or (batch, num_bins+1)
        batch_indices = jnp.arange(x.shape[0])
        return param[batch_indices, bin_idx if param_bins == "bin" else bin_idx]

    # Gather the bin-specific parameters.
    # For each sample, get:
    #   x_left: lower x-bound for the bin,
    #   y_bottom: lower y-bound for the bin,
    #   width and height for the bin,
    #   left and right derivatives.
    batch_idx = jnp.arange(x.shape[0])
    x_left = x_bins[batch_idx, bin_idx]
    y_bottom = y_bins[batch_idx, bin_idx]
    bin_width = widths[batch_idx, bin_idx]
    bin_height = heights[batch_idx, bin_idx]
    d_left = derivatives[batch_idx, bin_idx]
    d_right = derivatives[batch_idx, bin_idx + 1]

    # Compute normalized position in the bin.
    theta = (x - x_left) / bin_width
    theta = jnp.clip(theta, 0.0, 1.0)

    # For this simplified version, we use a convex (but learnable) combination:
    # forward transform (if inverse==False):
    #   y = y_bottom + bin_height * theta
    # with constant log_det = log(bin_height/bin_width)
    # (A full rational quadratic spline would use a nonlinear function of theta.)
    if not inverse:
        y = y_bottom + bin_height * theta
        log_det = jnp.log(bin_height) - jnp.log(bin_width)
        return y, log_det
    else:
        # Inverse transform: given y, recover theta, then x.
        # First, find the bin for y (using y_bins).
        bin_idx_y = jnp.sum(y[..., None] >= y_bins, axis=-1) - 1
        bin_idx_y = jnp.clip(bin_idx_y, 0, num_bins - 1)
        x_left = x_bins[batch_idx, bin_idx_y]
        y_bottom = y_bins[batch_idx, bin_idx_y]
        bin_width = widths[batch_idx, bin_idx_y]
        bin_height = heights[batch_idx, bin_idx_y]
        theta = (y - y_bottom) / bin_height
        theta = jnp.clip(theta, 0.0, 1.0)
        x = x_left + bin_width * theta
        log_det = -(jnp.log(bin_height) - jnp.log(bin_width))
        return x, log_det


# ----------------------------
# 4. Conditional Spline Module (Combining the conditioner and spline transform)
# ----------------------------
class ConditionalSpline(nn.Module):
    num_bins: int
    hidden_dim: int
    left: float = -3.0
    right: float = 3.0
    bottom: float = -3.0
    top: float = 3.0

    @nn.compact
    def __call__(self, x, context, inverse=False):
        # x: (batch, 1) ; context: (batch, context_dim)
        widths, heights, derivatives = SplineConditioner(
            num_bins=self.num_bins, hidden_dim=self.hidden_dim
        )(context)
        # Squeeze x to shape (batch,)
        x = jnp.squeeze(x, axis=-1)
        y, log_det = rational_quadratic_spline(
            x,
            widths,
            heights,
            derivatives,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
            inverse=inverse,
        )
        # Return y with shape (batch, 1) and log_det (batch,)
        return y[:, None], log_det


# ----------------------------
# 5. Overall Transformer + Spline NPE Model
# ----------------------------
class TransformerSplineNPE(nn.Module):
    obs_dim: int  # dimension of observation (per token)
    model_dim: int  # Transformer model dimension
    num_layers: int  # Number of Transformer layers
    num_heads: int  # Number of attention heads
    num_bins: int  # Number of bins for spline
    spline_hidden_dim: int  # Hidden dimension for spline conditioner

    @nn.compact
    def __call__(self, obs, theta, inverse=False):
        # obs: (batch, seq_len, obs_dim)
        # theta: (batch, 1)
        context = TransformerEncoder(
            model_dim=self.model_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )(obs)
        # Apply the conditional spline transform.
        transformed, log_det = ConditionalSpline(
            num_bins=self.num_bins, hidden_dim=self.spline_hidden_dim
        )(theta, context, inverse=inverse)
        return transformed, log_det

    def log_prob(self, params, obs, theta):
        # In our flow model we assume that theta = f(z; context)
        # where z comes from a base distribution (standard normal).
        # To evaluate log p(theta|obs) we invert the flow: z = f^{-1}(theta; context)
        z, log_det = self.apply({"params": params}, obs, theta, inverse=True)
        # Base log probability for z ~ N(0,1) (per dimension).
        log_pz = -0.5 * (jnp.log(2 * jnp.pi) + z**2)
        log_pz = jnp.sum(log_pz, axis=-1)
        return log_pz + log_det

    def sample(self, params, obs, rng, num_samples=1):
        # Sample from the posterior: first sample z ~ N(0,1), then invert the flow.
        batch_size = obs.shape[0]
        z = jax.random.normal(rng, (batch_size * num_samples, 1))
        # Get context from obs.
        context = TransformerEncoder(
            model_dim=self.model_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )(obs)
        # Repeat context for each sample.
        context_rep = jnp.repeat(context, num_samples, axis=0)
        # Invert the transform: given base sample z, compute theta.
        theta_sample, _ = ConditionalSpline(
            num_bins=self.num_bins, hidden_dim=self.spline_hidden_dim
        )(z, context_rep, inverse=True)
        # Reshape to (batch, num_samples, 1)
        theta_sample = theta_sample.reshape(batch_size, num_samples, 1)
        return theta_sample


# ----------------------------
# 6. Training Setup
# ----------------------------
def create_train_state(rng, model, learning_rate, obs_shape, theta_shape):
    dummy_obs = jnp.ones(obs_shape)
    dummy_theta = jnp.ones(theta_shape)
    params = model.init(rng, dummy_obs, dummy_theta)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def loss_fn(params, model, obs, theta):
    logp = model.log_prob(params, obs, theta)
    return -jnp.mean(logp)


@jax.jit
def train_step(state, model, obs, theta):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, model, obs, theta)
    state = state.apply_gradients(grads=grads)
    return state, loss


# ----------------------------
# 7. Simulator & Data Generation
# ----------------------------
def simulator(theta, noise_std=0.1, rng_key=None):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    noise = noise_std * jax.random.normal(rng_key, theta.shape)
    return theta + noise


# Generate synthetic data.
n_samples = 10000
theta_true = jnp.linspace(-3, 3, n_samples).reshape(n_samples, 1)
# Use a fixed RNG key for data simulation.
rng_sim = jax.random.PRNGKey(42)
obs = simulator(theta_true, noise_std=0.1, rng_key=rng_sim)
# For the transformer, add a sequence dimension (seq_len=1).
obs = obs[:, None, :]  # shape: (n_samples, 1, obs_dim)

# ----------------------------
# 8. Instantiate Model & Train
# ----------------------------
# Hyperparameters
obs_dim = 1
model_dim = 64
num_layers = 2
num_heads = 4
num_bins = 8
spline_hidden_dim = 32
learning_rate = 1e-3
batch_size = 128
n_epochs = 20

# Instantiate our model.
model = TransformerSplineNPE(
    obs_dim=obs_dim,
    model_dim=model_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    num_bins=num_bins,
    spline_hidden_dim=spline_hidden_dim,
)

# Create a training state.
rng = jax.random.PRNGKey(0)
state = create_train_state(
    rng, model, learning_rate, (batch_size, 1, obs_dim), (batch_size, 1)
)

# Convert data to NumPy for shuffling.
obs_np = np.array(obs)
theta_np = np.array(theta_true)

# Training loop.
n_train = obs_np.shape[0]
train_losses = []
for epoch in range(n_epochs):
    perm = np.random.permutation(n_train)
    epoch_loss = 0
    for i in range(0, n_train, batch_size):
        batch_idx = perm[i : i + batch_size]
        batch_obs = jnp.array(obs_np[batch_idx])
        batch_theta = jnp.array(theta_np[batch_idx])
        state, loss = train_step(state, model, batch_obs, batch_theta)
        epoch_loss += loss.item() * batch_obs.shape[0]
    epoch_loss /= n_train
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

# Plot training loss.
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Negative Log-Likelihood")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()

# ----------------------------
# 9. Posterior Sampling Example
# ----------------------------
rng_sample = jax.random.PRNGKey(100)
# Take a test observation (here, a single observation)
test_obs = jnp.array([[[0.5]]])  # shape: (1, 1, obs_dim)
samples = model.sample(state.params, test_obs, rng_sample, num_samples=100)
print("Posterior samples for observation 0.5:", samples.squeeze())

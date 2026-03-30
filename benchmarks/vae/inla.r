# ============================================================
# Batch runner (with intercept prior/posterior export)
# - true lengthscale ℓ ∈ {10, 30, 50}
# - grid sizes N ∈ {256, 576, 1024, 2304, 4096} (sides 16, 32, 48, 64)
# - simulate (σ² fixed to 1), spatial masking, fit with SAME priors
# - export lengthscale & intercept prior/posterior + plots
# - save per-run artifacts and a master summary.csv
# ============================================================

library(INLA)
library(ggplot2)
library(dplyr)
library(patchwork)
library(mvtnorm) # Added for direct MVN sampling

# ----------------------------
# 0) Global experiment config
# ----------------------------
generate <- FALSE   # < FLAG: set FALSE to check reproducibility only
lengthscales <- c(10, 30, 50)       # true ℓ
grid_sizes   <- c(256, 576, 1024, 2304, 4096)    # total cells (must be perfect squares)
obs_ratio    <- 0.6
mask_method  <- "grf"        # "grf" or "discs"

# Priors used for ALL runs
prior_mu2 <- -3.0     # θ2 = log κ ~ N(prior_mu2, prior_sd2^2)
prior_sd2 <-0.8
beta0_prior_mean <- 0.0  # intercept β0 ~ N(beta0_prior_mean, beta0_prior_sd^2)
beta0_prior_sd   <- sqrt(1000)

# Masking parameters
grf_range   <- 70
grf_sigma   <- 1
n_discs     <- 6
disc_radius <- 12

n_samp_len  <- 5000
out_root    <- "results/INLA_raw_batch_results"
dir.create(out_root, showWarnings = FALSE, recursive = TRUE)

# =========================
# 1) Core helper functions
# =========================
# ---- simulate or load GP–Poisson with FIXED ℓ, σ²=1 ----
generate_gp_poisson_data_fixed_ell <- function(grid_side,
                                               lengthscale_true,
                                               intercept = 1.5,
                                               seed = 123,
                                               prior_mu2 = -3.0,
                                               prior_sd2 = 0.8,
                                               generate = FALSE) {
  set.seed(seed)

  n_grid <- grid_side^2
  x_coords <- seq(0, 100, length.out = grid_side)
  y_coords <- seq(0, 100, length.out = grid_side)
  grid_data <- expand.grid(x = x_coords, y = y_coords)
  grid_data$loc_id <- seq_len(nrow(grid_data))

  # Mesh spacing
  h <- 100 / grid_side
  max_edge_inner <- 0.75 * h
  max_edge_outer <- 1.5 * h
  cutoff <- h * 0.5
  buffer <- 0.5 * 100

  # Construct mesh and SPDE
  mesh <- inla.mesh.2d(
    loc = as.matrix(grid_data[, c("x","y")]),
    max.edge = c(max_edge_inner, max_edge_outer),
    cutoff = cutoff,
    offset = c(buffer, buffer)
  )
  if (is.null(colnames(mesh$loc))) colnames(mesh$loc) <- c("x","y","z")

  spde <- inla.spde2.matern(mesh = mesh, alpha = 1.5, constr = TRUE)
  A <- inla.spde.make.A(mesh = mesh, loc = as.matrix(grid_data[, c("x","y")]))

  if (!generate) {
    # Load previous masked_y_and_fitted.csv
    load_file <- file.path("results/INLA_raw_batch_results",
                           sprintf("ell_%02d", as.integer(lengthscale_true)),
                           sprintf("n_%d", n_grid),
                           "masked_y_and_fitted.csv")
    loaded <- read.csv(load_file, stringsAsFactors = FALSE)

    # Use loaded x, y, observed_full, mask_obs, observed
    grid_data$x <- loaded$x
    grid_data$y <- loaded$y
    grid_data$mask_obs <- as.logical(loaded$mask_obs)
    grid_data$observed_full <- as.numeric(loaded$observed_full)
    grid_data$observed      <- as.numeric(loaded$observed)

    # Reconstruct latent spatial effect and linear predictor
    spatial_effect <- grid_data$observed_full - mean(grid_data$observed_full, na.rm = TRUE)
    eta <- intercept + spatial_effect
    lambda <- pmin(exp(eta), 1e6)
    grid_data$spatial_effect <- spatial_effect
    grid_data$linear_predictor <- eta
    grid_data$expected_lambda <- lambda

  } else {
    # Full simulation
    d_grid_mat <- as.matrix(dist(grid_data[, c("x","y")]))
    C_grid <- exp(-d_grid_mat / lengthscale_true)
    diag(C_grid) <- diag(C_grid) + 5e-4
    spatial_effect <- as.vector(rmvnorm(n = 1, mean = rep(0, n_grid), sigma = C_grid, method = 'chol'))

    eta <- intercept + spatial_effect
    lambda <- pmin(exp(eta), 1e6)
    observed <- rpois(n = n_grid, lambda = lambda)

    grid_data$observed_full <- observed
    grid_data$observed <- observed
    grid_data$mask_obs <- rep(TRUE, n_grid)  # initially all observed
    grid_data$spatial_effect <- spatial_effect
    grid_data$linear_predictor <- eta
    grid_data$expected_lambda <- lambda
  }

  # SPDE-related parameters
  kappa_true <- 1 / lengthscale_true
  sigma_true <- 1.0
  tau_true <- 1 / (sigma_true * sqrt(4 * pi) * kappa_true)
  theta1_true <- log(tau_true)
  theta2_true <- log(kappa_true)

  list(
    data = grid_data,
    mesh = mesh,
    spde = spde,
    A = A,
    true_spatial = grid_data$spatial_effect,
    hyperparameters = list(
      theta1 = theta1_true,
      theta2 = theta2_true,
      tau = tau_true,
      kappa = kappa_true,
      range = lengthscale_true,
      sigma = sigma_true,
      lengthscale_true = lengthscale_true,
      intercept_true = intercept,
      priors = list(
        theta2_normal = list(mu = prior_mu2, sd = prior_sd2),
        theta1_determined = "-0.5*log(4*pi) - theta2  (fixes sigma^2=1)"
      )
    )
  )
}

# ---- spatial masking (GRF threshold or discs) ----
generate_obs_mask_spatial <- function(sim_result,
                                      obs_ratio = 0.6,
                                      method = c("grf", "discs"),
                                      seed = 2024,
                                      grf_range = 70, grf_sigma = 1,
                                      n_discs = 6, disc_radius = 12) {
  method <- match.arg(method)
  set.seed(seed)

  d    <- sim_result$data
  mesh <- sim_result$mesh
  A    <- sim_result$A
  n    <- nrow(d)

  if (method == "grf") {
    spde_mask <- inla.spde2.matern(mesh = mesh, alpha = 1.5, constr = FALSE)
    kappa_m <- sqrt(8) / grf_range
    tau_m   <- 1 / (grf_sigma * sqrt(4 * pi) * kappa_m)
    Qm      <- inla.spde2.precision(spde_mask, theta = c(log(tau_m), log(kappa_m)))
    u_m <- as.vector(inla.qsample(1, Qm, seed = seed + 1))
    z   <- as.vector(A %*% u_m)
    thr <- stats::quantile(z, probs = 1 - obs_ratio, na.rm = TRUE)
    mask_obs <- z >= thr         # TRUE = observed, FALSE = missing
    return(mask_obs)
  } else {
    centers <- sample.int(n, size = n_discs, replace = FALSE)
    mask_obs <- rep(TRUE, n)
    for (i in centers) {
      dx <- d$x - d$x[i]; dy <- d$y - d$y[i]
      miss_idx <- (dx*dx + dy*dy) <= disc_radius^2
      mask_obs[miss_idx] <- FALSE
    }
    target_missing <- 1 - obs_ratio
    missing_frac   <- mean(!mask_obs)
    if (missing_frac > 0 && abs(missing_frac - target_missing) > 0.05) {
      scale <- sqrt(target_missing / max(missing_frac, 1e-6))
      mask_obs <- rep(TRUE, n)
      r2 <- (disc_radius * scale)^2
      for (i in centers) {
        dx <- d$x - d$x[i]; dy <- d$y - d$y[i]
        miss_idx <- (dx*dx + dy*dy) <= r2
        mask_obs[miss_idx] <- FALSE
      }
    }
    return(mask_obs)
  }
}

apply_mask_spatial <- function(sim_result,
                               obs_ratio = 0.6,
                               method = c("grf", "discs"),
                               seed = 2024,
                               grf_range = 70, grf_sigma = 1,
                               n_discs = 6, disc_radius = 12,
                               generate = FALSE) {   # <-- add generate flag
  d <- sim_result$data

  if (!generate) {
    # Use pre-saved mask
    mask_obs <- d$mask_obs
  } else {
    mask_obs <- generate_obs_mask_spatial(sim_result, obs_ratio, method, seed,
                                          grf_range, grf_sigma, n_discs, disc_radius)
  }
  d$mask_obs <- mask_obs
  d$observed[!mask_obs] <- NA_real_
  sim_result$data <- d
  sim_result
}

# ---- plotting masked data ----
plot_masked_data <- function(sim_result, title = "Observed data with masking (grey = missing)") {
  d <- sim_result$data
  ggplot() +
    geom_tile(data = d[!d$mask_obs, ],
              aes(x = x, y = y),
              fill = "grey90", color = "white", linewidth = 0.1) +
    geom_tile(data = d[d$mask_obs, ],
              aes(x = x, y = y, fill = observed),
              color = "white", linewidth = 0.1) +
    scale_fill_viridis_c(name = "count", na.value = "grey90") +
    coord_equal() +
    theme_minimal() +
    labs(title = title, x = "x", y = "y") +
    theme(panel.grid = element_blank())
}
# ---- Reproducibility police ----

save_or_check_reproducibility <- function(dat_to_save, out_file, ell, N, generate = FALSE, tol = 1e-8) {
  # Ensure only relevant columns are compared
  cols_check <- c("observed", "observed_full", "mask_obs")

  if (generate) {
    return()
  } else {
    if (!file.exists(out_file)) {
      stop(sprintf("File %s does not exist. Cannot check reproducibility.", out_file))
    }

    old <- read.csv(out_file, stringsAsFactors = FALSE)[, cols_check, drop = FALSE]
    dat_to_save <- dat_to_save[, cols_check, drop = FALSE]

    # Force proper types
    old$mask_obs <- as.logical(old$mask_obs)
    dat_to_save$mask_obs <- as.logical(dat_to_save$mask_obs)

    old$observed_full <- as.numeric(old$observed_full)
    dat_to_save$observed_full <- as.numeric(dat_to_save$observed_full)

    old$observed <- as.numeric(old$observed)
    dat_to_save$observed <- as.numeric(dat_to_save$observed)

    # Check for differences
    diffs <- list()
    for (col in cols_check) {
      a <- dat_to_save[[col]]
      b <- old[[col]]

      if (is.numeric(a) && is.numeric(b)) {
        mismatches <- sum(abs(a - b) > tol, na.rm = TRUE)
        if (mismatches > 0) diffs[[col]] <- mismatches
      } else if (is.logical(a) && is.logical(b)) {
        mismatches <- sum(a != b, na.rm = TRUE)
        if (mismatches > 0) diffs[[col]] <- mismatches
      } else {
        stop(sprintf("Column %s has unsupported type for reproducibility check", col))
      }
    }

    if (length(diffs) > 0) {
      msg <- paste(sprintf("%s: %d mismatches", names(diffs), unlist(diffs)), collapse = "; ")
      stop(sprintf("Reproducibility check failed for ell=%g, N=%d. %s", ell, N, msg))
    } else {
      message(sprintf("Reproducibility check passed for ell=%g, N=%d", ell, N))
    }
  }
}


# ---- fit with SAME priors (Gaussian on θ2; σ²≈1 constraint; explicit β0 prior) ----
fit_inla_with_same_priors <- function(sim_result,
                                      eps_sum_sd = 0.05,
                                      beta0_prior_mean = 0.0,
                                      beta0_prior_sd   = sqrt(1000)) {
  d    <- sim_result$data
  A    <- sim_result$A
  spde <- sim_result$spde

  mu2 <- sim_result$hyperparameters$priors$theta2_normal$mu
  sd2 <- sim_result$hyperparameters$priors$theta2_normal$sd

  C    <- 0.5 * log(4 * pi)
  mu   <- c(-C - mu2,  mu2)
  v    <- sd2^2
  s2   <- eps_sum_sd^2
  cov12 <- -v + s2/2
  Sigma <- matrix(c(v, cov12, cov12, v), nrow = 2, byrow = TRUE)
  Q     <- solve(Sigma)

  spde_prior <- spde
  spde_prior$param.inla$theta.mu <- mu
  spde_prior$param.inla$theta.Q  <- Q

  stk <- inla.stack(
    data = list(y = d$observed),    # NA for masked
    A    = list(A, 1),
    effects = list(spatial = 1:spde_prior$n.spde,
                   intercept = rep(1, nrow(d))),   # fixed effect called 'intercept'
    tag = "obs"
  )
  formula <- y ~ -1 + intercept + f(spatial, model = spde_prior)

  t_run <- system.time({
    fit <- inla(formula,
                family = "poisson",
                data   = inla.stack.data(stk),
                control.fixed = list(
                  mean = list(intercept = beta0_prior_mean),
                  prec = list(intercept = 1/(beta0_prior_sd^2))
                ),
                control.predictor = list(A = inla.stack.A(stk),
                                         compute = TRUE,
                                         link = 1),  # log-link for NA rows
                control.compute    = list(config = TRUE),
                control.inla = list(strategy = "laplace", int.strategy = "grid"),
                verbose = FALSE)
  })

  list(fit = fit, stack = stk, runtime = t_run,
       prior = list(mu = mu, Sigma = Sigma, Q = Q, C = C,
                    mu2 = mu2, sd2 = sd2,
                    beta0_mean = beta0_prior_mean, beta0_sd = beta0_prior_sd))
}

# ---- find θ2 marginal ----
get_theta2_marginal <- function(fit) {
  nms <- names(fit$marginals.hyperpar)
  pick <- grep("^Theta2|[Kk]appa|Range", nms, value = TRUE)
  if (length(pick) == 0 && length(nms) >= 2) pick <- nms[2]
  fit$marginals.hyperpar[[pick[1]]]
}

# ---- find intercept marginal ----
get_intercept_marginal <- function(fit) {
  nms <- names(fit$marginals.fixed)
  pick <- nms[match("intercept", nms)]
  if (is.na(pick)) {
    pick <- grep("intercept", nms, ignore.case = TRUE, value = TRUE)[1]
  }
  fit$marginals.fixed[[pick]]
}

# =========================
# 2) Batch loop over runs
# =========================
summary_rows <- list()
run_id <- 0L

for (ell in lengthscales) {
  for (N in grid_sizes) {
    side <- as.integer(round(sqrt(N)))
    if (side * side != N) stop("grid_sizes must be perfect squares. Got: ", N)

    run_id <- run_id + 1L
    set.seed(1000 + run_id)

    # --- simulate with fixed ℓ ---
    sim <- generate_gp_poisson_data_fixed_ell(
      grid_side      = side,
      lengthscale_true = ell,
      intercept      = 1.5,
      seed           = 1000 + run_id,
      prior_mu2      = prior_mu2,
      prior_sd2      = prior_sd2,
      generate = generate
    )

    # --- apply spatial masking ---
    sim <- apply_mask_spatial(sim,
                              obs_ratio  = obs_ratio,
                              method     = mask_method,
                              seed       = 2000 + run_id,
                              grf_range  = grf_range,
                              grf_sigma  = grf_sigma,
                              n_discs    = n_discs,
                              disc_radius= disc_radius,
                              generate=generate)

    # --- plot masked data (for saving later) ---
    plt_mask <- plot_masked_data(sim,
                                 title = sprintf("Masked data (ℓ_true=%g, N=%d)", ell, N))

    # --- fit with same priors ---
    fit_out <- fit_inla_with_same_priors(sim,
                                         eps_sum_sd = 0.05,
                                         beta0_prior_mean = beta0_prior_mean,
                                         beta0_prior_sd   = beta0_prior_sd)

    # --- runtime ---
    runtime_sec <- as.numeric(fit_out$runtime["elapsed"])
    cat(sprintf("[Run %02d] ℓ_true=%g, N=%d (side=%d)  runtime=%.3fs\n",
                run_id, ell, N, side, runtime_sec))

    # --- lengthscale prior & posterior ---
    set.seed(3000 + run_id)
    theta2_prior_samp <- rnorm(n_samp_len, mean = prior_mu2, sd = prior_sd2)
    l_prior <- exp(-theta2_prior_samp)

    marg_theta2_post <- get_theta2_marginal(fit_out$fit)
    marg_l_post <- inla.tmarginal(function(t) exp(-t), marg_theta2_post)
    set.seed(4000 + run_id)
    l_post <- inla.rmarginal(n_samp_len, marg_l_post)

    df_len <- rbind(
      transform(as.data.frame(density(l_prior)[c("x","y")]), what = "Prior"),
      transform(as.data.frame(density(l_post)[c("x","y")]),  what = "Posterior")
    )
    plt_len <- ggplot(df_len, aes(x, y, color = what)) +
      geom_line(linewidth = 1) +
      geom_vline(xintercept = ell, linetype = "dashed") +
      labs(title = sprintf("Lengthscale ℓ (prior vs posterior), ℓ_true=%g", ell),
           x = "ℓ", y = "Density", color = NULL) +
      theme_minimal()

    # --- intercept prior & posterior ---
    set.seed(5000 + run_id)
    b0_prior <- rnorm(n_samp_len,
                      mean = fit_out$prior$beta0_mean,
                      sd   = fit_out$prior$beta0_sd)
    marg_b0_post <- get_intercept_marginal(fit_out$fit)
    set.seed(6000 + run_id)
    b0_post <- inla.rmarginal(n_samp_len, marg_b0_post)

    df_b0 <- rbind(
      transform(as.data.frame(density(b0_prior)[c("x","y")]), what = "Prior"),
      transform(as.data.frame(density(b0_post)[c("x","y")]),  what = "Posterior")
    )
    plt_b0 <- ggplot(df_b0, aes(x, y, color = what)) +
      geom_line(linewidth = 1) +
      geom_vline(xintercept = sim$hyperparameters$intercept_true, linetype = "dashed") +
      labs(title = sprintf("Intercept β₀ (prior vs posterior), ℓ_true=%g", ell),
           x = "β₀", y = "Density", color = NULL) +
      theme_minimal()

    # --- predictions (ŷ) aligned to observation indices ---
    idx_obs <- inla.stack.index(fit_out$stack, "obs")$data
    y_hat <- fit_out$fit$summary.fitted.values$mean[idx_obs]
    sim$data$y_hat <- y_hat

    # --- observed vs fitted plot ---
    lims <- range(c(sim$data$observed, sim$data$y_hat), na.rm = TRUE)
    p_obs <- ggplot() +
      geom_tile(data = sim$data[!sim$data$mask_obs, ],
                aes(x = x, y = y),
                fill = "grey90", color = "white", linewidth = 0.1) +
      geom_tile(data = sim$data[sim$data$mask_obs, ],
                aes(x = x, y = y, fill = observed),
                color = "white", linewidth = 0.1) +
      scale_fill_viridis_c(limits = lims, name = "count") +
      coord_equal() +
      theme_minimal() +
      labs(title = "Observed (masked in grey)", x = "x", y = "y") +
      theme(panel.grid = element_blank())

    p_fit <- ggplot(sim$data, aes(x = x, y = y, fill = y_hat)) +
      geom_tile(color = "white", linewidth = 0.1) +
      scale_fill_viridis_c(limits = lims, name = "count") +
      coord_equal() +
      theme_minimal() +
      labs(title = "Fitted (E[y|data])", x = "x", y = "y") +
      theme(panel.grid = element_blank())

    plt_of <- p_obs + p_fit + plot_layout(ncol = 2) +
      plot_annotation(title = sprintf("Observed vs Fitted  (ℓ_true=%g, N=%d)", ell, N))

    # --- run-level output folder ---
    run_dir <- file.path(out_root,
                         sprintf("ell_%02d", as.integer(ell)),
                         sprintf("n_%d", N))
    dir.create(run_dir, showWarnings = FALSE, recursive = TRUE)

    # --- save artifacts ---
    out_file <- file.path(run_dir, "masked_y_and_fitted.csv")
    dat_to_save <- sim$data[, c("loc_id","x","y","observed_full","observed","mask_obs","y_hat")]
    save_or_check_reproducibility(dat_to_save, out_file, ell, N, generate = generate)
    write.csv(dat_to_save, file.path(run_dir, "masked_y_and_fitted.csv"), row.names = FALSE)
    write.csv(data.frame(lengthscale_prior = l_prior),
              file.path(run_dir, "lengthscale_prior_samples.csv"), row.names = FALSE)
    write.csv(data.frame(lengthscale_posterior = l_post),
              file.path(run_dir, "lengthscale_posterior_samples.csv"), row.names = FALSE)
    write.csv(data.frame(intercept_prior = b0_prior),
              file.path(run_dir, "intercept_prior_samples.csv"), row.names = FALSE)
    write.csv(data.frame(intercept_posterior = b0_post),
              file.path(run_dir, "intercept_posterior_samples.csv"), row.names = FALSE)


    ggsave(file.path(run_dir, "lengthscale_prior_vs_posterior.png"),
           plot = plt_len, width = 7, height = 5, dpi = 150)
    ggsave(file.path(run_dir, "intercept_prior_vs_posterior.png"),
           plot = plt_b0, width = 7, height = 5, dpi = 150)
    ggsave(file.path(run_dir, "observed_vs_fitted_masked.png"),
           plot = plt_of, width = 10, height = 5, dpi = 150)
    ggsave(file.path(run_dir, "masked_data_only.png"),
           plot = plt_mask, width = 7, height = 5, dpi = 150)

    writeLines(sprintf("elapsed_seconds: %.3f", runtime_sec),
               con = file.path(run_dir, "runtime.txt"))

    # --- summary row ---
    len_q <- quantile(l_post, c(0.025, 0.5, 0.975))
    b0_q  <- quantile(b0_post, c(0.025, 0.5, 0.975))
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      run_id = run_id,
      ell_true = ell,
      grid_size = N,
      grid_side = side,
      prior_mu2 = prior_mu2,
      prior_sd2 = prior_sd2,
      beta0_prior_mean = beta0_prior_mean,
      beta0_prior_sd   = beta0_prior_sd,
      beta0_true       = sim$hyperparameters$intercept_true,
      ell_post_mean    = mean(l_post),
      ell_post_median  = as.numeric(len_q["50%"]),
      ell_post_q025    = as.numeric(len_q["2.5%"]),
      ell_post_q975    = as.numeric(len_q["97.5%"]),
      beta0_post_mean   = mean(b0_post),
      beta0_post_median = as.numeric(b0_q["50%"]),
      beta0_post_q025   = as.numeric(b0_q["2.5%"]),
      beta0_post_q975   = as.numeric(b0_q["97.5%"]),
      runtime_sec = runtime_sec,
      stringsAsFactors = FALSE
    )
  }
}

# Master summary
summary_tbl <- bind_rows(summary_rows)
write.csv(summary_tbl, file.path(out_root, "summary.csv"), row.names = FALSE)

cat("All runs completed. Summary written to", file.path(out_root, "summary.csv"), "\n")

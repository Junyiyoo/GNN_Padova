"""Prepare centre and ordered neighbour inputs for the directional models."""

import numpy as np


def scale_directional_inputs(x, y, fit_mask, neighbors):
    """Return scaled centre features, ordered features, targets and scaling statistics."""
    # Fit on training nodes for tuning, then on train + validation for the final fit.
    fit_x = x[fit_mask].astype(np.float64)
    x_mean = fit_x.mean(axis=0)
    x_std = fit_x.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    center = ((x - x_mean) / x_std).astype(np.float32)

    # Keep each direction in its own slot. A missing neighbour has a zero block
    # and a zero presence flag, so it can be distinguished from an average neighbour.
    present = neighbors >= 0
    blocks = np.zeros((*neighbors.shape, center.shape[1]), dtype=np.float32)
    blocks[present] = center[neighbors[present]]
    ordered = np.concatenate([center, blocks.reshape(len(x), -1), present.astype(np.float32)], axis=1)

    fit_y = y[fit_mask].astype(np.float64)
    y_mean, y_std = float(fit_y.mean()), float(fit_y.std())
    if y_std < 1e-8:
        raise ValueError("The fitting nodes have insufficient LST variation for scaling.")
    scaler = dict(x_mean=x_mean.tolist(), x_std=x_std.tolist(), y_mean=y_mean, y_std=y_std)
    return center, ordered, (y - y_mean) / y_std, scaler

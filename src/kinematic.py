from __future__ import annotations

import numpy as np

from src.config import CRPRConfig


def radial_clip(vector: np.ndarray, radius: float, epsilon: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= radius:
        return vector.copy()
    scale = radius / max(norm, epsilon)
    return vector * scale


def kinematics_within_limits(path: np.ndarray, cfg: CRPRConfig, atol: float = 1e-9) -> bool:
    if len(path) <= 1:
        return True

    velocities = np.diff(path, axis=0) / cfg.dt
    if np.any(np.linalg.norm(velocities, axis=1) > cfg.v_max + atol):
        return False

    if len(velocities) <= 1:
        return True

    accelerations = np.diff(velocities, axis=0) / cfg.dt
    return not np.any(np.linalg.norm(accelerations, axis=1) > cfg.a_max + atol)


def bidirectional_kinematic_projection(
    path: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    cfg: CRPRConfig,
) -> np.ndarray:
    """
    Paper-aligned double-pass projection:
    1) forward velocity clipping
    2) backward acceleration clipping

    We additionally pin both endpoints to preserve the planning problem.
    """
    out = path.copy()
    out[0] = start.copy()
    out[-1] = goal.copy()

    if len(out) <= 2:
        return out

    # Forward pass: enforce velocity bounds.
    velocities = np.zeros((len(out) - 1, out.shape[1]), dtype=float)
    for k in range(len(out) - 1):
        v_k = (out[k + 1] - out[k]) / cfg.dt
        v_k_clipped = radial_clip(v_k, cfg.v_max, cfg.epsilon)
        velocities[k] = v_k_clipped
        if k + 1 < len(out) - 1:
            out[k + 1] = out[k] + v_k_clipped * cfg.dt

    # Re-pin goal before backward pass.
    out[0] = start.copy()
    out[-1] = goal.copy()

    # Backward pass: enforce acceleration bounds.
    # We propagate corrections backward through the interior velocities.
    for k in range(len(velocities) - 1, 0, -1):
        a_k = (velocities[k] - velocities[k - 1]) / cfg.dt
        a_k_clipped = radial_clip(a_k, cfg.a_max, cfg.epsilon)
        velocities[k] = velocities[k - 1] + a_k_clipped * cfg.dt

    # Reconstruct interior points from corrected velocities, preserving both ends as much as possible.
    recon = np.zeros_like(out)
    recon[0] = start.copy()
    for k in range(len(velocities)):
        recon[k + 1] = recon[k] + velocities[k] * cfg.dt

    # Blend reconstruction with fixed goal by linearly distributing endpoint drift.
    drift = goal - recon[-1]
    if len(recon) > 1:
        for k in range(len(recon)):
            alpha = k / (len(recon) - 1)
            recon[k] = recon[k] + alpha * drift

    recon[0] = start.copy()
    recon[-1] = goal.copy()
    return recon
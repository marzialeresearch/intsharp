"""
Interface sharpening source terms.

Implements:
- PM (Parameswaran-Mandal)
- CL (Chiu-Lin, isotropic diffusion form)
- Olsson-Kreiss (CLS [2] 2007, anisotropic diffusion)
- ACLS (Desjardins et al. [5] 2008, algebraic phi_inv normal)
- CLS 2010 ([3], non-conservative with mapped normal)
- LCLS 2012 ([7], localized with beta = 4 psi(1-psi))
- LCLS 2014 ([4], variable pseudo-time localization)
- CLS 2015 ([10], mapping function phi_map)
- CLS 2017 ([11], inverse transform with cosh)
- SCLS (Chiodi-Desjardins [8] 2018, self-correcting)

Applied post-step via operator splitting.
Supports both 1D and 2D.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .boundary import BoundaryCondition
from .registry import register_sharpening


# ---------------------------------------------------------------------------
# 1D Gradient/Divergence Helpers
# ---------------------------------------------------------------------------

def _grad_periodic(f: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Central difference gradient with periodic BC (1D)."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)


def _div_periodic(q: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Central difference divergence with periodic BC (1D)."""
    return (np.roll(q, -1) - np.roll(q, 1)) / (2 * dx)


def _grad_nonperiodic(
    f: NDArray[np.float64],
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Central difference gradient with non-periodic BC (1D)."""
    grad = np.zeros_like(f)
    # Interior: central difference
    grad[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    # Boundaries: one-sided
    grad[0] = (f[1] - f[0]) / dx
    grad[-1] = (f[-1] - f[-2]) / dx
    return grad


# ---------------------------------------------------------------------------
# 2D Gradient/Divergence Helpers
# ---------------------------------------------------------------------------

def _grad_periodic_2d(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Central difference gradient with periodic BC (2D).
    
    Returns (df/dx, df/dy).
    """
    # df/dx: roll along axis=1 (x-direction)
    dfdx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    # df/dy: roll along axis=0 (y-direction)
    dfdy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dy)
    return dfdx, dfdy


def _div_periodic_2d(
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """
    Central difference divergence with periodic BC (2D).
    
    div(q) = dqx/dx + dqy/dy
    """
    dqx_dx = (np.roll(qx, -1, axis=1) - np.roll(qx, 1, axis=1)) / (2 * dx)
    dqy_dy = (np.roll(qy, -1, axis=0) - np.roll(qy, 1, axis=0)) / (2 * dy)
    return dqx_dx + dqy_dy


def _grad_nonperiodic_2d(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Central difference gradient with non-periodic BC (2D).
    
    Returns (df/dx, df/dy).
    """
    ny, nx = f.shape
    dfdx = np.zeros_like(f)
    dfdy = np.zeros_like(f)
    
    # df/dx: interior central, boundaries one-sided
    dfdx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
    dfdx[:, 0] = (f[:, 1] - f[:, 0]) / dx
    dfdx[:, -1] = (f[:, -1] - f[:, -2]) / dx
    
    # df/dy: interior central, boundaries one-sided
    dfdy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)
    dfdy[0, :] = (f[1, :] - f[0, :]) / dy
    dfdy[-1, :] = (f[-1, :] - f[-2, :]) / dy
    
    return dfdx, dfdy


def _div_nonperiodic_2d(
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Central difference divergence with non-periodic BC (2D).
    
    div(q) = dqx/dx + dqy/dy
    """
    ny, nx = qx.shape
    dqx_dx = np.zeros_like(qx)
    dqy_dy = np.zeros_like(qy)
    
    # dqx/dx
    dqx_dx[:, 1:-1] = (qx[:, 2:] - qx[:, :-2]) / (2 * dx)
    dqx_dx[:, 0] = (qx[:, 1] - qx[:, 0]) / dx
    dqx_dx[:, -1] = (qx[:, -1] - qx[:, -2]) / dx
    
    # dqy/dy
    dqy_dy[1:-1, :] = (qy[2:, :] - qy[:-2, :]) / (2 * dy)
    dqy_dy[0, :] = (qy[1, :] - qy[0, :]) / dy
    dqy_dy[-1, :] = (qy[-1, :] - qy[-2, :]) / dy
    
    return dqx_dx + dqy_dy


@register_sharpening("pm")
def pm_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Parameswaran-Mandal sharpening.

    RHS = -K * psi * (1-psi) * (1-2*psi) + eps * (1-2*psi) * |grad(psi)|

    where K = 1 / (4 * eps^2).

    Parameters
    ----------
    psi : NDArray
        Volume fraction field.
    dx : float
        Grid spacing.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness (epsilon).
    strength : float
        Sharpening strength multiplier (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    # Compute gradient
    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi)

    # PM sharpening coefficient
    K = 1.0 / (4.0 * eps_target**2)

    # RHS
    rhs = (
        -K * psi * (1 - psi) * (1 - 2*psi)
        + eps_target * (1 - 2*psi) * abs_grad
    )

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


@register_sharpening("cl")
def cl_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Chiu-Lin sharpening.

    RHS = div(eps * grad(psi) - psi * (1-psi) * n_hat)

    where n_hat = grad(psi) / |grad(psi)|

    Parameters
    ----------
    psi : NDArray
        Volume fraction field.
    dx : float
        Grid spacing.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness.
    strength : float
        Sharpening strength multiplier.
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    eta = 1e-12  # Small number to avoid division by zero

    # Compute gradient
    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi) + eta

    # Unit normal (sign of gradient)
    n_hat = grad_psi / abs_grad

    # Flux: eps * grad(psi) - psi * (1-psi) * n_hat
    flux = eps_target * grad_psi - psi * (1 - psi) * n_hat

    # Divergence of flux
    if bc.bc_type == "periodic":
        rhs = _div_periodic(flux, dx)
    else:
        # Central difference for interior, one-sided for boundaries
        rhs = np.zeros_like(psi)
        rhs[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
        rhs[0] = (flux[1] - flux[0]) / dx
        rhs[-1] = (flux[-1] - flux[-2]) / dx

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


# Calibrated PM constant from dissertation (200 * 1.97715965626)
C_PM_CALIBRATED = 395.43193125


@register_sharpening("pm_cal")
def pm_sharpening_calibrated(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Parameswaran-Mandal sharpening with calibrated constant (1D).

    RHS = -C_PM * psi * (1-psi) * (1-2*psi) + eps * (1-2*psi) * |grad(psi)|

    where C_PM ≈ 395.4 is a calibrated constant (independent of eps).

    Parameters
    ----------
    psi : NDArray
        Volume fraction field.
    dx : float
        Grid spacing.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness (epsilon).
    strength : float
        Sharpening strength multiplier (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    # Compute gradient
    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi)

    # RHS with calibrated constant
    rhs = (
        -C_PM_CALIBRATED * psi * (1 - psi) * (1 - 2*psi)
        + eps_target * (1 - 2*psi) * abs_grad
    )

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2D Sharpening Methods
# ---------------------------------------------------------------------------


@register_sharpening("pm_2d")
def pm_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Parameswaran-Mandal sharpening (2D).

    RHS = -K * psi * (1-psi) * (1-2*psi) + eps * (1-2*psi) * |grad(psi)|

    where K = 1 / (4 * eps^2) and |grad| = sqrt((dpsi/dx)^2 + (dpsi/dy)^2).

    Parameters
    ----------
    psi : NDArray
        Volume fraction field (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness (epsilon).
    strength : float
        Sharpening strength multiplier (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    eta = 1e-12  # Small number to avoid division by zero

    # Compute gradient
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    # Gradient magnitude
    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + eta

    # PM sharpening coefficient
    K = 1.0 / (4.0 * eps_target**2)

    # RHS
    rhs = (
        -K * psi * (1 - psi) * (1 - 2*psi)
        + eps_target * (1 - 2*psi) * abs_grad
    )

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


@register_sharpening("pm_cal_2d")
def pm_sharpening_calibrated_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Parameswaran-Mandal sharpening with calibrated constant (2D).

    RHS = -C_PM * psi * (1-psi) * (1-2*psi) + eps * (1-2*psi) * |grad(psi)|

    where C_PM ≈ 395.4 is a calibrated constant (independent of eps).
    |grad| = sqrt((dpsi/dx)^2 + (dpsi/dy)^2).

    Parameters
    ----------
    psi : NDArray
        Volume fraction field (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness (epsilon).
    strength : float
        Sharpening strength multiplier (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    eta = 1e-12  # Small number to avoid division by zero

    # Compute gradient
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    # Gradient magnitude
    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + eta

    # RHS with calibrated constant
    rhs = (
        -C_PM_CALIBRATED * psi * (1 - psi) * (1 - 2*psi)
        + eps_target * (1 - 2*psi) * abs_grad
    )

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


@register_sharpening("cl_2d")
def cl_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Chiu-Lin sharpening (2D).

    RHS = div(eps * grad(psi) - psi * (1-psi) * n_hat)

    where n_hat = grad(psi) / |grad(psi)|.

    Parameters
    ----------
    psi : NDArray
        Volume fraction field (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness.
    strength : float
        Sharpening strength multiplier.
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    eta = 1e-12  # Small number to avoid division by zero

    # Compute gradient
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    # Gradient magnitude
    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + eta

    # Unit normal vector
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    # Flux components: eps * grad(psi) - psi * (1-psi) * n_hat
    flux_x = eps_target * dfdx - psi * (1 - psi) * nx
    flux_y = eps_target * dfdy - psi * (1 - psi) * ny

    # Divergence of flux
    if bc.bc_type == "periodic":
        rhs = _div_periodic_2d(flux_x, flux_y, dx, dy)
    else:
        rhs = _div_nonperiodic_2d(flux_x, flux_y, dx, dy, bc)

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Shared Utilities for Literature Methods
# ---------------------------------------------------------------------------

_ETA = 1e-12


def _phi_inv(psi: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    """Algebraic signed-distance inversion: phi = eps * ln(psi / (1-psi))."""
    psi_c = np.clip(psi, _ETA, 1.0 - _ETA)
    return eps * np.log(psi_c / (1.0 - psi_c))


def _laplacian_periodic(f: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Second-order Laplacian with periodic BC (1D)."""
    return (np.roll(f, -1) - 2.0 * f + np.roll(f, 1)) / (dx * dx)


def _laplacian_nonperiodic(
    f: NDArray[np.float64],
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Second-order Laplacian with non-periodic BC (1D)."""
    lap = np.zeros_like(f)
    lap[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / (dx * dx)
    lap[0] = (f[1] - 2.0 * f[0] + f[1]) / (dx * dx)
    lap[-1] = (f[-2] - 2.0 * f[-1] + f[-2]) / (dx * dx)
    return lap


def _laplacian_periodic_2d(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """Second-order Laplacian with periodic BC (2D)."""
    d2f_dx2 = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dx * dx)
    d2f_dy2 = (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dy * dy)
    return d2f_dx2 + d2f_dy2


def _laplacian_nonperiodic_2d(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Second-order Laplacian with non-periodic BC (2D)."""
    lap = np.zeros_like(f)
    # x-direction
    lap[:, 1:-1] += (f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]) / (dx * dx)
    lap[:, 0] += (f[:, 1] - 2.0 * f[:, 0] + f[:, 1]) / (dx * dx)
    lap[:, -1] += (f[:, -2] - 2.0 * f[:, -1] + f[:, -2]) / (dx * dx)
    # y-direction
    lap[1:-1, :] += (f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]) / (dy * dy)
    lap[0, :] += (f[1, :] - 2.0 * f[0, :] + f[1, :]) / (dy * dy)
    lap[-1, :] += (f[-2, :] - 2.0 * f[-1, :] + f[-2, :]) / (dy * dy)
    return lap


def _div_nonperiodic(
    q: NDArray[np.float64],
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Central difference divergence with non-periodic BC (1D)."""
    d = np.zeros_like(q)
    d[1:-1] = (q[2:] - q[:-2]) / (2 * dx)
    d[0] = (q[1] - q[0]) / dx
    d[-1] = (q[-1] - q[-2]) / dx
    return d


# ---------------------------------------------------------------------------
# CLS [2] (2007) — Olsson-Kreiss
# ---------------------------------------------------------------------------


@register_sharpening("olsson_kreiss")
def olsson_kreiss_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Olsson-Kreiss CLS reinitialization (1D).

    RHS = div[eps (grad psi . n) n  -  psi(1-psi) n]

    Normal n is frozen at the start of the sub-step (n = sign(grad psi)).
    In 1D the anisotropic diffusion reduces to isotropic, so this is
    equivalent to CL with a frozen normal.
    """
    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi) + _ETA
    n = grad_psi / abs_grad

    # Flux: eps * (grad_psi . n) * n  -  psi*(1-psi)*n
    #   In 1D: (grad_psi . n) = |grad_psi|, so flux = eps*|grad_psi|*n - psi*(1-psi)*n
    flux = eps_target * abs_grad * n - psi * (1.0 - psi) * n

    if bc.bc_type == "periodic":
        rhs = _div_periodic(flux, dx)
    else:
        rhs = _div_nonperiodic(flux, dx, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("olsson_kreiss_2d")
def olsson_kreiss_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Olsson-Kreiss CLS reinitialization (2D).

    RHS = div[eps (grad psi . n) n  -  psi(1-psi) n]

    Normal n is frozen at the start of the sub-step.
    Diffusion is anisotropic (only along the interface normal).
    """
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    # grad psi . n  (directional derivative along n)
    grad_dot_n = dfdx * nx + dfdy * ny

    # Flux components: eps * (grad psi . n) * n_i  -  psi*(1-psi) * n_i
    compressive = psi * (1.0 - psi)
    flux_x = eps_target * grad_dot_n * nx - compressive * nx
    flux_y = eps_target * grad_dot_n * ny - compressive * ny

    if bc.bc_type == "periodic":
        rhs = _div_periodic_2d(flux_x, flux_y, dx, dy)
    else:
        rhs = _div_nonperiodic_2d(flux_x, flux_y, dx, dy, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ACLS [5] (2008) — Desjardins et al.
# ---------------------------------------------------------------------------


@register_sharpening("acls")
def acls_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Accurate Conservative Level Set reinitialization (1D).

    Same flux form as Olsson-Kreiss, but normal n is computed from the
    algebraic signed-distance inversion phi_inv = eps * ln(psi/(1-psi)).
    """
    phi = _phi_inv(psi, eps_target)

    if bc.bc_type == "periodic":
        grad_phi = _grad_periodic(phi, dx)
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_phi = _grad_nonperiodic(phi, dx, bc)
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad_phi = np.abs(grad_phi) + _ETA
    n = grad_phi / abs_grad_phi

    grad_dot_n = grad_psi * n
    flux = eps_target * grad_dot_n * n - psi * (1.0 - psi) * n

    if bc.bc_type == "periodic":
        rhs = _div_periodic(flux, dx)
    else:
        rhs = _div_nonperiodic(flux, dx, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("acls_2d")
def acls_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Accurate Conservative Level Set reinitialization (2D).

    Normal n from phi_inv = eps * ln(psi/(1-psi));
    flux = eps (grad psi . n) n  -  psi(1-psi) n.
    """
    phi = _phi_inv(psi, eps_target)

    if bc.bc_type == "periodic":
        dphi_dx, dphi_dy = _grad_periodic_2d(phi, dx, dy)
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dphi_dx, dphi_dy = _grad_nonperiodic_2d(phi, dx, dy, bc)
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2) + _ETA
    nx = dphi_dx / abs_grad_phi
    ny = dphi_dy / abs_grad_phi

    grad_dot_n = dfdx * nx + dfdy * ny
    compressive = psi * (1.0 - psi)
    flux_x = eps_target * grad_dot_n * nx - compressive * nx
    flux_y = eps_target * grad_dot_n * ny - compressive * ny

    if bc.bc_type == "periodic":
        rhs = _div_periodic_2d(flux_x, flux_y, dx, dy)
    else:
        rhs = _div_nonperiodic_2d(flux_x, flux_y, dx, dy, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CLS [3] (2010) — Non-conservative form with mapped normal
# ---------------------------------------------------------------------------


@register_sharpening("cls_2010")
def cls_2010_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [3] reinitialization (1D).

    RHS = n . grad[eps |grad psi| - psi(1-psi)]

    Normal n from mapped field phi(psi) = psi^alpha / (psi^alpha + (1-psi)^alpha).
    """
    alpha = kwargs.get("mapping_alpha", 2.0)

    psi_c = np.clip(psi, _ETA, 1.0 - _ETA)
    phi_map = psi_c**alpha / (psi_c**alpha + (1.0 - psi_c)**alpha)

    if bc.bc_type == "periodic":
        grad_phi = _grad_periodic(phi_map, dx)
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_phi = _grad_nonperiodic(phi_map, dx, bc)
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad_phi = np.abs(grad_phi) + _ETA
    n = grad_phi / abs_grad_phi

    # Scalar quantity: eps*|grad psi| - psi*(1-psi)
    scalar = eps_target * np.abs(grad_psi) - psi * (1.0 - psi)

    # Gradient of scalar
    if bc.bc_type == "periodic":
        grad_scalar = _grad_periodic(scalar, dx)
    else:
        grad_scalar = _grad_nonperiodic(scalar, dx, bc)

    rhs = n * grad_scalar

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("cls_2010_2d")
def cls_2010_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [3] reinitialization (2D).

    RHS = n . grad[eps |grad psi| - psi(1-psi)]

    Normal n from mapped field phi(psi) = psi^alpha / (psi^alpha + (1-psi)^alpha).
    """
    alpha = kwargs.get("mapping_alpha", 2.0)

    psi_c = np.clip(psi, _ETA, 1.0 - _ETA)
    phi_map = psi_c**alpha / (psi_c**alpha + (1.0 - psi_c)**alpha)

    if bc.bc_type == "periodic":
        dphi_dx, dphi_dy = _grad_periodic_2d(phi_map, dx, dy)
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dphi_dx, dphi_dy = _grad_nonperiodic_2d(phi_map, dx, dy, bc)
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2) + _ETA
    nx = dphi_dx / abs_grad_phi
    ny = dphi_dy / abs_grad_phi

    # Scalar: eps*|grad psi| - psi*(1-psi)
    abs_grad_psi = np.sqrt(dfdx**2 + dfdy**2)
    scalar = eps_target * abs_grad_psi - psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        ds_dx, ds_dy = _grad_periodic_2d(scalar, dx, dy)
    else:
        ds_dx, ds_dy = _grad_nonperiodic_2d(scalar, dx, dy, bc)

    rhs = nx * ds_dx + ny * ds_dy

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# LCLS [7] (2012) — Localized CLS
# ---------------------------------------------------------------------------


@register_sharpening("lcls_2012")
def lcls_2012_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Localized CLS reinitialization (1D).

    RHS = beta * eps * laplacian(psi)  -  beta * div(psi(1-psi) n)

    Localization: beta = 4 psi (1-psi), which peaks at the interface
    and vanishes in bulk regions.
    """
    beta = 4.0 * psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
        lap_psi = _laplacian_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)
        lap_psi = _laplacian_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi) + _ETA
    n = grad_psi / abs_grad

    # Compressive flux
    comp_flux = psi * (1.0 - psi) * n

    if bc.bc_type == "periodic":
        div_comp = _div_periodic(comp_flux, dx)
    else:
        div_comp = _div_nonperiodic(comp_flux, dx, bc)

    rhs = beta * eps_target * lap_psi - beta * div_comp

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("lcls_2012_2d")
def lcls_2012_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Localized CLS reinitialization (2D).

    RHS = beta * eps * laplacian(psi)  -  beta * div(psi(1-psi) n)

    beta = 4 psi (1-psi).
    """
    beta = 4.0 * psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
        lap_psi = _laplacian_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)
        lap_psi = _laplacian_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    compressive = psi * (1.0 - psi)
    flux_x = compressive * nx
    flux_y = compressive * ny

    if bc.bc_type == "periodic":
        div_comp = _div_periodic_2d(flux_x, flux_y, dx, dy)
    else:
        div_comp = _div_nonperiodic_2d(flux_x, flux_y, dx, dy, bc)

    rhs = beta * eps_target * lap_psi - beta * div_comp

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# LCLS [4] (2014) — Variable pseudo-time localization
# ---------------------------------------------------------------------------


@register_sharpening("lcls_2014")
def lcls_2014_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    LCLS [4] reinitialization with variable pseudo-time (1D).

    RHS = a_tilde * div[eps (grad psi . n) n  -  psi(1-psi) n]

    Simplified localization weight a_tilde = 4 psi (1-psi).
    """
    a_tilde = 4.0 * psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi) + _ETA
    n = grad_psi / abs_grad

    grad_dot_n = grad_psi * n
    flux = eps_target * grad_dot_n * n - psi * (1.0 - psi) * n

    if bc.bc_type == "periodic":
        div_flux = _div_periodic(flux, dx)
    else:
        div_flux = _div_nonperiodic(flux, dx, bc)

    rhs = a_tilde * div_flux

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("lcls_2014_2d")
def lcls_2014_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    LCLS [4] reinitialization with variable pseudo-time (2D).

    RHS = a_tilde * div[eps (grad psi . n) n  -  psi(1-psi) n]

    a_tilde = 4 psi (1-psi).
    """
    a_tilde = 4.0 * psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    grad_dot_n = dfdx * nx + dfdy * ny
    compressive = psi * (1.0 - psi)
    flux_x = eps_target * grad_dot_n * nx - compressive * nx
    flux_y = eps_target * grad_dot_n * ny - compressive * ny

    if bc.bc_type == "periodic":
        div_flux = _div_periodic_2d(flux_x, flux_y, dx, dy)
    else:
        div_flux = _div_nonperiodic_2d(flux_x, flux_y, dx, dy, bc)

    rhs = a_tilde * div_flux

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CLS [10] (2015) — Mapping function approach
# ---------------------------------------------------------------------------


@register_sharpening("cls_2015")
def cls_2015_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [10] mapping-function reinitialization (1D).

    RHS = div[psi(1-psi) (|grad phi_map| - 1) n_Gamma]

    phi_map = (psi+eps)^gamma / ((psi+eps)^gamma + (1-psi+eps)^gamma)
    n_Gamma = grad(phi_map) / |grad(phi_map)|
    """
    gamma = kwargs.get("mapping_gamma", 2.0)
    eps = eps_target

    psi_c = np.clip(psi, 0.0, 1.0)
    a = (psi_c + eps)**gamma
    b = (1.0 - psi_c + eps)**gamma
    phi_map = a / (a + b)

    if bc.bc_type == "periodic":
        grad_phi = _grad_periodic(phi_map, dx)
    else:
        grad_phi = _grad_nonperiodic(phi_map, dx, bc)

    abs_grad_phi = np.abs(grad_phi) + _ETA
    n = grad_phi / abs_grad_phi

    flux = psi * (1.0 - psi) * (abs_grad_phi - 1.0) * n

    if bc.bc_type == "periodic":
        rhs = _div_periodic(flux, dx)
    else:
        rhs = _div_nonperiodic(flux, dx, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("cls_2015_2d")
def cls_2015_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [10] mapping-function reinitialization (2D).

    RHS = div[psi(1-psi) (|grad phi_map| - 1) n_Gamma]

    phi_map = (psi+eps)^gamma / ((psi+eps)^gamma + (1-psi+eps)^gamma)
    """
    gamma = kwargs.get("mapping_gamma", 2.0)
    eps = eps_target

    psi_c = np.clip(psi, 0.0, 1.0)
    a = (psi_c + eps)**gamma
    b = (1.0 - psi_c + eps)**gamma
    phi_map = a / (a + b)

    if bc.bc_type == "periodic":
        dphi_dx, dphi_dy = _grad_periodic_2d(phi_map, dx, dy)
    else:
        dphi_dx, dphi_dy = _grad_nonperiodic_2d(phi_map, dx, dy, bc)

    abs_grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2) + _ETA
    nx = dphi_dx / abs_grad_phi
    ny = dphi_dy / abs_grad_phi

    coeff = psi * (1.0 - psi) * (abs_grad_phi - 1.0)
    flux_x = coeff * nx
    flux_y = coeff * ny

    if bc.bc_type == "periodic":
        rhs = _div_periodic_2d(flux_x, flux_y, dx, dy)
    else:
        rhs = _div_nonperiodic_2d(flux_x, flux_y, dx, dy, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CLS [11] (2017) — Inverse transform with cosh
# ---------------------------------------------------------------------------


@register_sharpening("cls_2017")
def cls_2017_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [11] inverse-transform reinitialization (1D).

    RHS = div[ 1/(4 cosh^2(phi_inv / (2 eps)))  *  (|grad phi_inv . n| - 1)  *  n ]

    phi_inv = eps * ln(psi/(1-psi)),  n = grad(phi_inv) / |grad(phi_inv)|.
    """
    eps = eps_target
    phi = _phi_inv(psi, eps)

    if bc.bc_type == "periodic":
        grad_phi = _grad_periodic(phi, dx)
    else:
        grad_phi = _grad_nonperiodic(phi, dx, bc)

    abs_grad_phi = np.abs(grad_phi) + _ETA
    n = grad_phi / abs_grad_phi

    # 1 / (4 cosh^2(phi / (2 eps)))
    arg = np.clip(phi / (2.0 * eps), -50.0, 50.0)
    weight = 1.0 / (4.0 * np.cosh(arg)**2)

    # |grad phi . n| = |grad phi| in 1D
    flux = weight * (abs_grad_phi - 1.0) * n

    if bc.bc_type == "periodic":
        rhs = _div_periodic(flux, dx)
    else:
        rhs = _div_nonperiodic(flux, dx, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("cls_2017_2d")
def cls_2017_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [11] inverse-transform reinitialization (2D).

    RHS = div[ 1/(4 cosh^2(phi_inv / (2 eps)))  *  (|grad phi_inv . n| - 1)  *  n ]

    phi_inv = eps * ln(psi/(1-psi)),  n = grad(phi_inv) / |grad(phi_inv)|.
    """
    eps = eps_target
    phi = _phi_inv(psi, eps)

    if bc.bc_type == "periodic":
        dphi_dx, dphi_dy = _grad_periodic_2d(phi, dx, dy)
    else:
        dphi_dx, dphi_dy = _grad_nonperiodic_2d(phi, dx, dy, bc)

    abs_grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2) + _ETA
    nx = dphi_dx / abs_grad_phi
    ny = dphi_dy / abs_grad_phi

    arg = np.clip(phi / (2.0 * eps), -50.0, 50.0)
    weight = 1.0 / (4.0 * np.cosh(arg)**2)

    # |grad phi . n| = |grad phi| since n is the unit gradient direction
    coeff = weight * (abs_grad_phi - 1.0)
    flux_x = coeff * nx
    flux_y = coeff * ny

    if bc.bc_type == "periodic":
        rhs = _div_periodic_2d(flux_x, flux_y, dx, dy)
    else:
        rhs = _div_nonperiodic_2d(flux_x, flux_y, dx, dy, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# SCLS [8] (2018) — Self-Correcting Level Set
# ---------------------------------------------------------------------------


@register_sharpening("scls")
def scls_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Self-Correcting Level Set reinitialization (1D).

    RHS = -div(psi(1-psi) m) + div(eps (grad psi . m) m) + div((1 - |m|^2) eps grad psi)

    m = eps grad(psi) / sqrt(eps^2 |grad psi|^2  +  alpha^2 exp(-beta eps^2 |grad psi|^2))
    """
    alpha_sc = kwargs.get("scls_alpha", 1e-3)
    beta_sc = kwargs.get("scls_beta", 1e3)
    eps = eps_target

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    grad_sq = grad_psi**2
    eps_grad_sq = eps**2 * grad_sq

    denom = np.sqrt(eps_grad_sq + alpha_sc**2 * np.exp(-beta_sc * eps_grad_sq)) + _ETA
    m = eps * grad_psi / denom
    m_sq = m**2

    # Term 1: -div(psi(1-psi) m)
    flux1 = psi * (1.0 - psi) * m
    # Term 2: div(eps (grad psi . m) m)
    grad_dot_m = grad_psi * m
    flux2 = eps * grad_dot_m * m
    # Term 3: div((1 - |m|^2) eps grad psi)
    flux3 = (1.0 - m_sq) * eps * grad_psi

    total_flux = -flux1 + flux2 + flux3

    if bc.bc_type == "periodic":
        rhs = _div_periodic(total_flux, dx)
    else:
        rhs = _div_nonperiodic(total_flux, dx, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("scls_2d")
def scls_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Self-Correcting Level Set reinitialization (2D).

    RHS = -div(psi(1-psi) m) + div(eps (grad psi . m) m) + div((1 - |m|^2) eps grad psi)

    m = eps grad(psi) / sqrt(eps^2 |grad psi|^2  +  alpha^2 exp(-beta eps^2 |grad psi|^2))
    """
    alpha_sc = kwargs.get("scls_alpha", 1e-3)
    beta_sc = kwargs.get("scls_beta", 1e3)
    eps = eps_target

    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    grad_sq = dfdx**2 + dfdy**2
    eps_grad_sq = eps**2 * grad_sq

    denom = np.sqrt(eps_grad_sq + alpha_sc**2 * np.exp(-beta_sc * eps_grad_sq)) + _ETA
    mx = eps * dfdx / denom
    my = eps * dfdy / denom
    m_sq = mx**2 + my**2

    # grad psi . m
    grad_dot_m = dfdx * mx + dfdy * my
    compressive = psi * (1.0 - psi)

    # Flux x = -psi(1-psi)*mx + eps*(grad.m)*mx + (1-|m|^2)*eps*dfdx
    fx = -compressive * mx + eps * grad_dot_m * mx + (1.0 - m_sq) * eps * dfdx
    fy = -compressive * my + eps * grad_dot_m * my + (1.0 - m_sq) * eps * dfdy

    if bc.bc_type == "periodic":
        rhs = _div_periodic_2d(fx, fy, dx, dy)
    else:
        rhs = _div_nonperiodic_2d(fx, fy, dx, dy, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)

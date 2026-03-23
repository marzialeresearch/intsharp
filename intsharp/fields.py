"""
Field containers with expression-based and image-based initial condition evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from .boundary import BoundaryCondition, create_bc
from .config import FieldConfig
from .domain import Domain1D, Domain2D, Domain


# ---------------------------------------------------------------------------
# Safe expression evaluation
# ---------------------------------------------------------------------------

# Whitelist of allowed names in IC expressions
SAFE_NAMESPACE: dict[str, Any] = {
    # Math functions
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "tanh": np.tanh,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "floor": np.floor,
    "ceil": np.ceil,
    "sign": np.sign,
    "heaviside": np.heaviside,
    # Constants
    "pi": np.pi,
    "e": np.e,
    # NumPy
    "np": np,
    # Builtins needed for expressions
    "min": np.minimum,
    "max": np.maximum,
}


def evaluate_expression_1d(
    expr: str,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Safely evaluate a math expression with 'x' as the spatial coordinate (1D).

    Parameters
    ----------
    expr : str
        Expression string (e.g., "0.5 * (1 + tanh((0.025 - abs(x)) / 0.002))").
    x : NDArray
        1D grid coordinates.

    Returns
    -------
    NDArray
        Evaluated values at each grid point.

    Raises
    ------
    ValueError
        If the expression is invalid or uses disallowed functions.
    """
    # Create namespace with x
    namespace = SAFE_NAMESPACE.copy()
    namespace["x"] = x

    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate initial condition expression: {expr!r}\n"
            f"Error: {e}"
        ) from e

    # Ensure result is an array of the right shape
    result = np.asarray(result, dtype=np.float64)
    if result.shape != x.shape:
        # Handle scalar expressions (constant IC)
        if result.ndim == 0:
            result = np.full_like(x, result)
        else:
            raise ValueError(
                f"IC expression result has shape {result.shape}, "
                f"expected {x.shape}"
            )

    return result


def evaluate_expression_2d(
    expr: str,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Safely evaluate a math expression with 'x', 'y', and 'r' as spatial coordinates (2D).

    Parameters
    ----------
    expr : str
        Expression string (e.g., "0.5 * (1 + tanh((0.15 - r) / 0.02))").
    X : NDArray
        2D meshgrid of x coordinates (shape: ny, nx).
    Y : NDArray
        2D meshgrid of y coordinates (shape: ny, nx).

    Returns
    -------
    NDArray
        Evaluated values at each grid point (shape: ny, nx).

    Raises
    ------
    ValueError
        If the expression is invalid or uses disallowed functions.
    """
    # Create namespace with x, y, and r
    namespace = SAFE_NAMESPACE.copy()
    namespace["x"] = X
    namespace["y"] = Y
    namespace["r"] = np.sqrt(X**2 + Y**2)

    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate initial condition expression: {expr!r}\n"
            f"Error: {e}"
        ) from e

    # Ensure result is an array of the right shape
    result = np.asarray(result, dtype=np.float64)
    if result.shape != X.shape:
        # Handle scalar expressions (constant IC)
        if result.ndim == 0:
            result = np.full_like(X, result)
        else:
            raise ValueError(
                f"IC expression result has shape {result.shape}, "
                f"expected {X.shape}"
            )

    return result


# ---------------------------------------------------------------------------
# Velocity expression evaluation
# ---------------------------------------------------------------------------

def evaluate_velocity_expression_1d(
    u_expr: str,
    x: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    """
    Evaluate a 1D velocity expression u(x, t).

    Parameters
    ----------
    u_expr : str
        Expression for u (e.g., "0.5", "sin(2*pi*x)").
    x : NDArray
        1D grid coordinates.
    t : float
        Current time.

    Returns
    -------
    NDArray
        Velocity values at each grid point (shape matches x).
    """
    namespace = SAFE_NAMESPACE.copy()
    namespace["x"] = x
    namespace["t"] = t

    try:
        result = eval(u_expr, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate velocity expression: {u_expr!r}\n"
            f"Error: {e}"
        ) from e

    result = np.asarray(result, dtype=np.float64)
    if result.shape != x.shape:
        if result.ndim == 0:
            result = np.full_like(x, result)
        else:
            raise ValueError(
                f"Velocity expression result has shape {result.shape}, "
                f"expected {x.shape}"
            )
    return result


def evaluate_velocity_expression_2d(
    u_expr: str,
    v_expr: str,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    t: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Evaluate 2D velocity expressions u(x, y, t) and v(x, y, t).

    Parameters
    ----------
    u_expr : str
        Expression for u component (e.g., "-2*pi*(y - 0.5)").
    v_expr : str
        Expression for v component (e.g., "2*pi*(x - 0.5)").
    X : NDArray
        2D meshgrid of x coordinates (shape: ny, nx).
    Y : NDArray
        2D meshgrid of y coordinates (shape: ny, nx).
    t : float
        Current time.

    Returns
    -------
    tuple[NDArray, NDArray]
        (u_arr, v_arr) velocity components at each grid point.
    """
    namespace = SAFE_NAMESPACE.copy()
    namespace["x"] = X
    namespace["y"] = Y
    namespace["r"] = np.sqrt(X**2 + Y**2)
    namespace["t"] = t

    # Evaluate u
    try:
        u_result = eval(u_expr, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate velocity.u expression: {u_expr!r}\n"
            f"Error: {e}"
        ) from e

    u_result = np.asarray(u_result, dtype=np.float64)
    if u_result.shape != X.shape:
        if u_result.ndim == 0:
            u_result = np.full_like(X, u_result)
        else:
            raise ValueError(
                f"velocity.u expression result has shape {u_result.shape}, "
                f"expected {X.shape}"
            )

    # Evaluate v
    try:
        v_result = eval(v_expr, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate velocity.v expression: {v_expr!r}\n"
            f"Error: {e}"
        ) from e

    v_result = np.asarray(v_result, dtype=np.float64)
    if v_result.shape != X.shape:
        if v_result.ndim == 0:
            v_result = np.full_like(X, v_result)
        else:
            raise ValueError(
                f"velocity.v expression result has shape {v_result.shape}, "
                f"expected {X.shape}"
            )

    return u_result, v_result


# ---------------------------------------------------------------------------
# Image-based initial condition
# ---------------------------------------------------------------------------

def load_image_ic(
    image_path: str | Path,
    domain: Domain,
    config_dir: Path | None = None,
) -> NDArray[np.float64]:
    """
    Load an image and convert it to a volume fraction field.

    The image is converted to grayscale, resized to match the domain grid,
    thresholded (dark pixels < 0.5 become 1, light pixels become 0),
    and a tanh-smoothed interface is applied.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file (PNG, JPG, etc.).
    domain : Domain1D or Domain2D
        The computational domain.
    config_dir : Path or None
        Directory of the config file for resolving relative paths.

    Returns
    -------
    NDArray[np.float64]
        Volume fraction field values. Shape is (n_points,) for 1D, (ny, nx) for 2D.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image-based initial conditions. "
            "Install with: pip install pillow"
        )

    # Resolve path relative to config file directory
    image_path = Path(image_path)
    if not image_path.is_absolute() and config_dir is not None:
        image_path = config_dir / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load and convert to grayscale
    img = Image.open(image_path).convert("L")

    # Determine target size
    if isinstance(domain, Domain2D):
        target_size = (domain.nx, domain.ny)  # PIL uses (width, height)
        dx = domain.dx
    else:
        target_size = (domain.nx, 1)
        dx = domain.dx

    # Resize using nearest neighbor to preserve sharp edges
    img_resized = img.resize(target_size, Image.Resampling.NEAREST)

    # Convert to numpy array and normalize to [0, 1]
    # PIL: row 0 = top of image, column 0 = left. Shape (height, width) = (ny, nx).
    img_array = np.array(img_resized, dtype=np.float64) / 255.0

    if isinstance(domain, Domain2D):
        # Align image with domain: row 0 = y_min (bottom), col 0 = x_min (left).
        # PIL: row 0 = image top, col 0 = image left.
        # Flip vertical so image top -> domain top (row ny-1).
        img_array = np.flipud(img_array)
    else:
        # For 1D, take the first row (or average if multiple rows)
        if img_array.ndim == 2:
            img_array = img_array.mean(axis=0)

    # Threshold: dark pixels (< 0.5) become 1 (alpha=1), light become 0
    binary_mask = (img_array < 0.5).astype(np.float64)

    # Compute signed distance field
    # Positive inside the mask (alpha=1 region), negative outside
    dist_inside = distance_transform_edt(binary_mask)
    dist_outside = distance_transform_edt(1.0 - binary_mask)
    signed_distance = dist_inside - dist_outside

    # Apply tanh profile with epsilon = 3 * dx
    eps = 3.0 * dx
    if eps > 0:
        alpha = 0.5 * (1.0 + np.tanh(signed_distance / eps))
    else:
        alpha = binary_mask

    return alpha.astype(np.float64)


# ---------------------------------------------------------------------------
# Field container
# ---------------------------------------------------------------------------

@dataclass
class Field:
    """
    A scalar field on the 1D or 2D domain.

    Attributes
    ----------
    name : str
        Field name.
    values : NDArray[np.float64]
        Current field values. Shape is (n_points,) for 1D, (ny, nx) for 2D.
    bc : BoundaryCondition
        Boundary condition for this field.
    sharpening_enabled : bool or None
        Per-field sharpening override. None means use global setting.
    sharpening_method : str or None
        Per-field sharpening method override. None means use global setting.
    """
    name: str
    values: NDArray[np.float64]
    bc: BoundaryCondition
    sharpening_enabled: bool | None = None
    sharpening_method: str | None = None

    def copy(self) -> "Field":
        """Return a copy of this field."""
        return Field(
            name=self.name,
            values=self.values.copy(),
            bc=self.bc,
            sharpening_enabled=self.sharpening_enabled,
            sharpening_method=self.sharpening_method,
        )


def create_field(
    config: FieldConfig,
    domain: Domain,
    config_dir: Path | None = None,
) -> Field:
    """
    Create a field from configuration and domain.

    Parameters
    ----------
    config : FieldConfig
        Field configuration.
    domain : Domain1D or Domain2D
        The computational domain.
    config_dir : Path or None
        Directory of the config file for resolving relative image paths.

    Returns
    -------
    Field
        Initialized field.
    """
    # Evaluate initial condition based on source type
    if config.initial_condition_image is not None:
        # Image-based initial condition
        values = load_image_ic(config.initial_condition_image, domain, config_dir)
    elif config.initial_condition is not None:
        # Expression-based initial condition
        if isinstance(domain, Domain2D):
            values = evaluate_expression_2d(config.initial_condition, domain.X, domain.Y)
        else:
            values = evaluate_expression_1d(config.initial_condition, domain.x)
    else:
        raise ValueError(
            f"Field '{config.name}' has no initial condition specified"
        )

    # Create boundary condition
    bc = create_bc(config.boundary)

    return Field(
        name=config.name,
        values=values,
        bc=bc,
        sharpening_enabled=config.sharpening,
        sharpening_method=config.sharpening_method,
    )


def create_fields(
    configs: list[FieldConfig],
    domain: Domain,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Create all fields from configuration.

    Parameters
    ----------
    configs : list[FieldConfig]
        List of field configurations.
    domain : Domain1D or Domain2D
        The computational domain.
    config_dir : Path or None
        Directory of the config file for resolving relative image paths.

    Returns
    -------
    dict[str, Field]
        Dictionary mapping field names to Field objects.
    """
    fields = {}
    for cfg in configs:
        fields[cfg.name] = create_field(cfg, domain, config_dir)
    return fields

"""
Configuration parsing and validation for the simulation framework.

Uses pydantic for strict schema validation with helpful error messages.

Supports two physics modes:
- advection_only: Volume fraction advection with optional sharpening (original)
- euler: Compressible Euler equations with AUSM+UP flux
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DomainConfig(BaseModel):
    """1D or 2D domain configuration."""
    x_min: float = Field(..., description="Left boundary of domain")
    x_max: float = Field(..., description="Right boundary of domain")
    n_points: Optional[int] = Field(None, gt=1, description="Number of grid points (1D only)")
    n_points_x: Optional[int] = Field(None, gt=1, description="Number of grid points in x (2D)")
    y_min: Optional[float] = Field(None, description="Bottom boundary of domain (2D)")
    y_max: Optional[float] = Field(None, description="Top boundary of domain (2D)")
    n_points_y: Optional[int] = Field(None, gt=1, description="Number of grid points in y (2D)")

    @model_validator(mode="after")
    def validate_domain_params(self) -> "DomainConfig":
        """Ensure either 1D or 2D parameters are specified."""
        is_2d = self.y_min is not None or self.y_max is not None or self.n_points_y is not None
        if is_2d:
            if self.y_min is None or self.y_max is None or self.n_points_y is None:
                raise ValueError(
                    "2D domain requires 'y_min', 'y_max', and 'n_points_y'"
                )
            if self.n_points_x is None and self.n_points is None:
                raise ValueError(
                    "2D domain requires 'n_points_x' (or 'n_points' for x)"
                )
        else:
            if self.n_points is None and self.n_points_x is None:
                raise ValueError("1D domain requires 'n_points'")
        return self

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return 2 if self.y_min is not None else 1

    @property
    def nx(self) -> int:
        """Number of points in x."""
        return self.n_points_x if self.n_points_x is not None else (self.n_points or 0)

    @property
    def ny(self) -> int:
        """Number of points in y (0 for 1D)."""
        return self.n_points_y if self.n_points_y is not None else 0

    @property
    def dx(self) -> float:
        """Grid spacing in x."""
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dy(self) -> float:
        """Grid spacing in y (0 for 1D)."""
        if self.ndim == 1:
            return 0.0
        return (self.y_max - self.y_min) / (self.ny - 1)  # type: ignore

    @property
    def L(self) -> float:
        """Domain length in x."""
        return self.x_max - self.x_min

    @property
    def Ly(self) -> float:
        """Domain length in y (0 for 1D)."""
        if self.ndim == 1:
            return 0.0
        return self.y_max - self.y_min  # type: ignore


class TimeConfig(BaseModel):
    """Time-stepping configuration."""
    dt: float = Field(..., gt=0, description="Time step size")
    n_steps: int = Field(..., gt=0, description="Number of time steps")

    @property
    def t_final(self) -> float:
        """Final simulation time."""
        return self.dt * self.n_steps


class BoundaryConfig(BaseModel):
    """Boundary condition configuration for a field."""
    type: Literal["periodic", "neumann", "dirichlet"] = Field(
        ..., description="Boundary condition type"
    )
    # For Neumann: specify gradient at boundaries
    gradient_left: Optional[float] = Field(
        None, description="Gradient at left boundary (Neumann)"
    )
    gradient_right: Optional[float] = Field(
        None, description="Gradient at right boundary (Neumann)"
    )
    # For Dirichlet: specify values at boundaries
    value_left: Optional[float] = Field(
        None, description="Value at left boundary (Dirichlet)"
    )
    value_right: Optional[float] = Field(
        None, description="Value at right boundary (Dirichlet)"
    )

    @model_validator(mode="after")
    def validate_bc_params(self) -> "BoundaryConfig":
        if self.type == "neumann":
            if self.gradient_left is None or self.gradient_right is None:
                raise ValueError(
                    "Neumann BC requires 'gradient_left' and 'gradient_right'"
                )
        elif self.type == "dirichlet":
            if self.value_left is None or self.value_right is None:
                raise ValueError(
                    "Dirichlet BC requires 'value_left' and 'value_right'"
                )
        return self


class FieldConfig(BaseModel):
    """Configuration for a single field (e.g., volume fraction)."""
    name: str = Field(..., description="Field name (e.g., 'alpha')")
    initial_condition: Optional[str] = Field(
        None, description="Expression for initial condition (uses 'x' for 1D, 'x', 'y', 'r' for 2D)"
    )
    initial_condition_image: Optional[str] = Field(
        None, description="Path to image file for initial condition (dark=1, light=0)"
    )
    boundary: BoundaryConfig = Field(..., description="Boundary condition")
    sharpening: Optional[bool] = Field(
        None,
        description="Enable sharpening for this field (overrides global setting if specified)"
    )
    sharpening_method: Optional[str] = Field(
        None,
        description="Per-field sharpening method override (e.g., 'pm', 'cl', 'olsson_kreiss'). Falls back to global sharpening.method if not set."
    )

    @model_validator(mode="after")
    def validate_ic_source(self) -> "FieldConfig":
        """Ensure exactly one of initial_condition or initial_condition_image is set."""
        has_expr = self.initial_condition is not None
        has_image = self.initial_condition_image is not None
        if has_expr and has_image:
            raise ValueError(
                "Field cannot have both 'initial_condition' and 'initial_condition_image'"
            )
        if not has_expr and not has_image:
            raise ValueError(
                "Field requires either 'initial_condition' or 'initial_condition_image'"
            )
        return self


class SolverConfig(BaseModel):
    """Spatial discretization (advection solver) configuration."""
    type: str = Field(..., description="Solver type (e.g., 'upwind')")


class TimeStepperConfig(BaseModel):
    """Time integration method configuration."""
    type: Literal["euler", "rk4"] = Field(
        "euler", description="Time-stepping method"
    )


class SharpeningConfig(BaseModel):
    """Interface sharpening configuration (optional)."""
    enabled: bool = Field(False, description="Enable sharpening")
    method: Literal[
        "pm", "pm_cal", "cl",
        "olsson_kreiss", "acls", "cls_2010",
        "lcls_2012", "lcls_2014",
        "cls_2015", "cls_2017", "scls",
    ] = Field("cl", description="Sharpening method")
    eps_target: float = Field(
        ..., gt=0, description="Target interface thickness"
    )
    strength: float = Field(1.0, gt=0, description="Sharpening strength (Gamma)")
    method_params: Optional[dict[str, Any]] = Field(
        None,
        description="Method-specific parameters (e.g., mapping_alpha, mapping_gamma, scls_alpha, scls_beta)"
    )


class SurfaceTensionConfig(BaseModel):
    """Surface tension diagnostic configuration (optional)."""
    enabled: bool = Field(False, description="Enable surface tension diagnostics")
    sigma: float = Field(0.07, gt=0, description="Surface tension coefficient")
    source_field: str = Field("alpha", description="Field to compute curvature from")
    smoothing_sigma: Optional[float] = Field(
        None, ge=0, description="Brackbill-style Gaussian smoothing sigma (grid cells). None = no smoothing."
    )
    interface_band_alpha_min: Optional[float] = Field(
        None, ge=0, le=1, description="Mask diagnostics to alpha >= this (raw alpha). None = no band."
    )
    interface_band_alpha_max: Optional[float] = Field(
        None, ge=0, le=1, description="Mask diagnostics to alpha <= this (raw alpha). None = no band."
    )

    @model_validator(mode="after")
    def validate_interface_band(self) -> "SurfaceTensionConfig":
        """Interface band requires both min and max."""
        has_min = self.interface_band_alpha_min is not None
        has_max = self.interface_band_alpha_max is not None
        if has_min != has_max:
            raise ValueError(
                "interface_band_alpha_min and interface_band_alpha_max must both be set or both None"
            )
        if has_min and self.interface_band_alpha_min >= self.interface_band_alpha_max:
            raise ValueError(
                "interface_band_alpha_min must be < interface_band_alpha_max"
            )
        return self


# ---------------------------------------------------------------------------
# Physics configuration (Euler mode)
# ---------------------------------------------------------------------------


class MaterialConfig(BaseModel):
    """Material properties for stiffened gas EOS."""
    name: Optional[str] = Field(None, description="Material name (e.g., 'water', 'air')")
    gamma: float = Field(..., gt=1, description="Heat capacity ratio (γ)")
    p_infinity: float = Field(0.0, ge=0, description="Stiffness parameter (p_∞). 0 for ideal gas.")
    rho_ref: Optional[float] = Field(None, gt=0, description="Reference density (for two-phase)")


class RiemannStateConfig(BaseModel):
    """State for one side of a Riemann problem."""
    rho: float = Field(..., gt=0, description="Density (mixture density for two-phase)")
    u: float = Field(..., description="Velocity")
    p: float = Field(..., gt=0, description="Pressure")
    alpha: Optional[float] = Field(
        None, ge=0, le=1, description="Volume fraction of phase 1 (for two-phase)"
    )


class EulerInitialConditionsConfig(BaseModel):
    """Initial conditions for Euler mode."""
    type: Literal["riemann", "uniform", "rti"] = Field(..., description="IC type")
    # Riemann problem (shock tube)
    x_discontinuity: Optional[float] = Field(
        None, description="Location of discontinuity (for riemann type)"
    )
    left: Optional[RiemannStateConfig] = Field(None, description="Left state (for riemann type)")
    right: Optional[RiemannStateConfig] = Field(None, description="Right state (for riemann type)")
    # Uniform state
    rho: Optional[float] = Field(None, gt=0, description="Uniform density (for uniform type)")
    u: Optional[float] = Field(None, description="Uniform velocity (for uniform type)")
    p: Optional[float] = Field(None, gt=0, description="Uniform pressure (for uniform type)")
    # 2D RTI setup (two-phase 5eq)
    interface_y0: Optional[float] = Field(
        None, description="Base y-location of perturbed interface (for rti type)"
    )
    alpha1_top: float = Field(
        0.0, ge=0, le=1, description="Top-region volume fraction of phase 1 (for rti type)"
    )
    alpha1_bottom: float = Field(
        1.0, ge=0, le=1, description="Bottom-region volume fraction of phase 1 (for rti type)"
    )
    perturbation_amplitude: float = Field(
        0.0, ge=0, description="Sinusoidal interface perturbation amplitude (for rti type)"
    )
    perturbation_mode_x: int = Field(
        1, ge=1, description="Interface perturbation mode number in x (for rti type)"
    )
    interface_thickness: float = Field(
        0.0, ge=0, description="Optional tanh interface thickness (for rti type)"
    )
    p0: float = Field(1.0e5, gt=0, description="Reference pressure at interface (for rti type)")
    u0: float = Field(0.0, description="Initial x-velocity (for rti type)")
    v0: float = Field(0.0, description="Initial y-velocity (for rti type)")
    v_perturbation_amplitude: float = Field(
        0.0,
        description="Optional additive y-velocity perturbation amplitude; interpreted as v = v0 + amp*cos(2*pi*mode*x/Lx) unless scaled by sound speed",
    )
    v_perturbation_mode_x: int = Field(
        1, ge=1, description="Mode number for optional y-velocity perturbation in x"
    )
    v_perturbation_scale_with_sound_speed: bool = Field(
        False, description="If true, scales perturbation by local sound speed c"
    )

    @model_validator(mode="after")
    def validate_ic_params(self) -> "EulerInitialConditionsConfig":
        """Validate IC parameters based on type."""
        if self.type == "riemann":
            if self.x_discontinuity is None:
                raise ValueError("Riemann IC requires 'x_discontinuity'")
            if self.left is None or self.right is None:
                raise ValueError("Riemann IC requires 'left' and 'right' states")
        elif self.type == "uniform":
            if self.rho is None or self.u is None or self.p is None:
                raise ValueError("Uniform IC requires 'rho', 'u', and 'p'")
        elif self.type == "rti":
            if self.interface_y0 is None:
                raise ValueError("RTI IC requires 'interface_y0'")
        return self


class EulerGravityConfig(BaseModel):
    """Optional gravity source for Euler mode."""
    enabled: bool = Field(False, description="Enable constant gravity source")
    gx: float = Field(0.0, description="Constant gravity acceleration in x")
    gy: float = Field(0.0, description="Constant gravity acceleration in y")


class PhysicsConfig(BaseModel):
    """Physics mode configuration."""
    mode: Literal["advection_only", "euler"] = Field(
        "advection_only",
        description="Physics mode: 'advection_only' (scalar advection) or 'euler' (compressible Euler)"
    )
    # Single-phase: material properties (for single-phase euler)
    material: Optional[MaterialConfig] = Field(
        None, description="Material properties (for single-phase euler)"
    )
    # Two-phase: phase1 and phase2 materials
    phase1: Optional[MaterialConfig] = Field(
        None, description="Phase 1 material (e.g., water). α=1 region."
    )
    phase2: Optional[MaterialConfig] = Field(
        None, description="Phase 2 material (e.g., air). α=0 region."
    )
    # Euler initial conditions (required for euler mode)
    euler_initial_conditions: Optional[EulerInitialConditionsConfig] = Field(
        None, description="Initial conditions for Euler mode"
    )
    # Euler boundary conditions
    euler_bc: Literal["transmissive", "reflective", "periodic"] = Field(
        "transmissive", description="Boundary condition for Euler mode"
    )
    euler_bc_x: Literal["transmissive", "reflective", "periodic"] = Field(
        "periodic", description="Boundary condition in x for 2D Euler mode"
    )
    euler_bc_y: Literal["transmissive", "reflective", "periodic"] = Field(
        "reflective", description="Boundary condition in y for 2D Euler mode"
    )
    gravity: EulerGravityConfig = Field(
        default_factory=EulerGravityConfig,
        description="Optional constant gravity source for Euler mode"
    )
    # Spatial reconstruction
    use_muscl: bool = Field(
        True, description="Use MUSCL reconstruction with Barth-Jespersen limiter (2nd order)"
    )
    # Spatial discretization for Euler mode
    euler_spatial_discretization: Literal["fv", "dg"] = Field(
        "fv",
        description="Euler spatial discretization: finite volume ('fv') or discontinuous Galerkin ('dg')."
    )
    # DG options (currently only P1 implemented)
    dg_order: int = Field(
        1, ge=1, le=3, description="DG polynomial order (supported: 1, 2, 3)."
    )
    dg_use_limiter: bool = Field(
        True, description="Enable DG slope limiter."
    )
    dg_use_positivity: bool = Field(
        True, description="Enable DG positivity scaling."
    )
    # Euler intercell flux calculator
    flux_calculator: Literal["ausm_plus_up", "hllc"] = Field(
        "ausm_plus_up",
        description="Intercell Riemann flux calculator for Euler mode."
    )
    # Two-phase model selection
    two_phase_model: Literal["mixture", "5eq"] = Field(
        "5eq",
        description="Two-phase model: 'mixture' (simplified) or '5eq' (5-equation model)"
    )

    @property
    def is_two_phase(self) -> bool:
        """Check if this is a two-phase simulation."""
        return self.phase1 is not None and self.phase2 is not None

    @model_validator(mode="after")
    def validate_euler_params(self) -> "PhysicsConfig":
        """Euler mode requires material(s) and initial conditions."""
        if self.mode == "euler":
            has_single = self.material is not None
            has_two = self.phase1 is not None and self.phase2 is not None
            if not has_single and not has_two:
                raise ValueError(
                    "Euler mode requires either 'physics.material' (single-phase) "
                    "or both 'physics.phase1' and 'physics.phase2' (two-phase)"
                )
            if has_single and has_two:
                raise ValueError(
                    "Specify either 'physics.material' OR 'physics.phase1/phase2', not both"
                )
            if self.euler_initial_conditions is None:
                raise ValueError("Euler mode requires 'physics.euler_initial_conditions'")
            if self.euler_spatial_discretization == "dg" and self.is_two_phase:
                raise ValueError(
                    "DG Euler is currently implemented for single-phase mode only."
                )
            if self.euler_initial_conditions.type == "rti" and not self.is_two_phase:
                raise ValueError("RTI IC currently requires two-phase Euler configuration.")
            if self.euler_initial_conditions.type == "rti" and self.two_phase_model != "5eq":
                raise ValueError("RTI IC currently requires two_phase_model: '5eq'.")
        return self


# ---------------------------------------------------------------------------
# Velocity configuration (constant or expression-based)
# ---------------------------------------------------------------------------

class ConstantVelocityConfig(BaseModel):
    """Constant velocity configuration."""
    type: Literal["constant"] = Field("constant", description="Velocity type")
    value: list[float] = Field(..., description="Constant velocity vector [u] (1D) or [u, v] (2D)")


class ExpressionVelocityConfig(BaseModel):
    """Expression-based velocity configuration (u(x,y,t), v(x,y,t))."""
    type: Literal["expression"] = Field("expression", description="Velocity type")
    u: str = Field(..., description="Expression for u component (can use x, y, t, r, pi, etc.)")
    v: Optional[str] = Field(None, description="Expression for v component (2D only)")


# Union type for velocity: constant or expression
VelocityConfig = Union[ConstantVelocityConfig, ExpressionVelocityConfig]


class CompareFieldConfig(BaseModel):
    """Configuration for a field in compare_fields (multi-field gif overlay)."""
    field: str = Field(..., description="Field name to plot")
    contour_levels: list[float] = Field(
        default_factory=lambda: [0.5],
        description="Contour levels to draw for 2D contour mode (default [0.5])"
    )
    color: Optional[str] = Field(None, description="Line color (auto-assigned if not specified)")
    linestyle: str = Field("-", description="Line style (e.g., '-', '--', ':')")


class MonitorConfig(BaseModel):
    """Output monitor configuration."""
    type: Literal["console", "png", "pdf", "svg", "gif", "mp4", "hdf5", "txt", "curve"] = Field(
        ..., description="Monitor type"
    )
    every_n_steps: Optional[int] = Field(
        None, gt=0, description="Output every N steps"
    )
    at_times: Optional[list[float]] = Field(
        None, description="Output at specific times"
    )
    # Single-field mode
    field: Optional[str] = Field(
        None, description="Field to output (for single-field gif/image/txt/curve)"
    )
    # Multi-field mode (for gif: overlays multiple fields)
    compare_fields: Optional[list[CompareFieldConfig]] = Field(
        None, description="Fields to compare (for multi-field gif overlay)"
    )
    # HDF5-specific
    fields: Optional[list[str]] = Field(
        None, description="Fields to output (for hdf5)"
    )
    # GIF style for 2D single-field mode
    style: Literal["pcolormesh", "contour"] = Field(
        "pcolormesh", description="2D single-field style: 'pcolormesh' (default) or 'contour'"
    )
    # Contour levels for 2D contour mode (single-field or compare_fields)
    contour_levels: list[float] = Field(
        default_factory=lambda: [0.5],
        description="Contour levels for 2D contour style (default [0.5])"
    )
    # Optional pcolormesh colormap (gif/mp4 2D): e.g. "viridis", "inferno", "plasma"
    colormap: Optional[str] = Field(
        None, description="Colormap for 2D pcolormesh (default viridis)"
    )
    # Optional contour overlay on pcolormesh: draw contour_levels in this color on top
    contour_overlay_color: Optional[str] = Field(
        None, description="If set, draw contour_levels on top of pcolormesh in this color"
    )
    # Contour-only mode (style=contour): contour line color (default blue)
    contour_color: Optional[str] = Field(
        None, description="Contour line color when style=contour (default blue)"
    )
    # Contour-only mode: axes background color (e.g. hex #2563eb)
    background_color: Optional[str] = Field(
        None, description="Axes background color when style=contour (e.g. hex #2563eb)"
    )
    # Image outputs (png, pdf, svg, gif, mp4): show colorbar for pcolormesh (default True)
    show_colorbar: Optional[bool] = Field(
        None, description="Show colorbar for pcolormesh (default True)"
    )
    # Image outputs: show annotations (ticks, tick labels, axis titles, plot title) (default True)
    show_annotations: Optional[bool] = Field(
        None, description="Show plot annotations: x/y ticks, tick labels, axis titles, plot title (default True)"
    )
    # Fixed color scale bounds for 2D pcolormesh gif/mp4
    vmin: Optional[float] = Field(
        None, description="Lower color bound for 2D pcolormesh (gif/mp4 single-field mode)"
    )
    vmax: Optional[float] = Field(
        None, description="Upper color bound for 2D pcolormesh (gif/mp4 single-field mode)"
    )
    # Quiver overlay (gif/mp4 2D pcolormesh): overlay vector field on top
    quiver_overlay_x: Optional[str] = Field(
        None, description="Field name for quiver x-component (e.g. csf_x). Requires quiver_overlay_y."
    )
    quiver_overlay_y: Optional[str] = Field(
        None, description="Field name for quiver y-component (e.g. csf_y). Requires quiver_overlay_x."
    )
    quiver_skip: Optional[int] = Field(
        None, gt=0, description="Plot every Nth quiver arrow for clarity (default 4)"
    )

    @model_validator(mode="after")
    def validate_quiver_overlay(self) -> "MonitorConfig":
        """Quiver overlay requires both x and y components."""
        has_x = self.quiver_overlay_x is not None
        has_y = self.quiver_overlay_y is not None
        if has_x != has_y:
            raise ValueError(
                "quiver_overlay_x and quiver_overlay_y must both be set or both None"
            )
        return self

    @model_validator(mode="after")
    def validate_output_trigger(self) -> "MonitorConfig":
        if self.every_n_steps is None and self.at_times is None:
            if self.type != "console":
                raise ValueError(
                    f"Monitor '{self.type}' requires 'every_n_steps' or 'at_times'"
                )
        return self

    @model_validator(mode="after")
    def validate_gif_field_mode(self) -> "MonitorConfig":
        """For gif/mp4: exactly one of field or compare_fields must be set."""
        if self.type in ("gif", "mp4"):
            has_field = self.field is not None
            has_compare = self.compare_fields is not None and len(self.compare_fields) > 0
            if has_field and has_compare:
                raise ValueError(f"{self.type} monitor: specify either 'field' or 'compare_fields', not both")
            if not has_field and not has_compare:
                raise ValueError(f"{self.type} monitor: requires 'field' or 'compare_fields'")
        return self

    @model_validator(mode="after")
    def validate_color_bounds(self) -> "MonitorConfig":
        if self.vmin is not None and self.vmax is not None and self.vmin >= self.vmax:
            raise ValueError("monitor.vmin must be < monitor.vmax")
        return self


class OutputConfig(BaseModel):
    """Output configuration."""
    directory: str = Field("./results", description="Output directory")
    monitors: list[MonitorConfig] = Field(
        default_factory=list, description="List of output monitors"
    )


class ConvergenceConfig(BaseModel):
    """Optional spatial-convergence study configuration."""
    enabled: bool = Field(False, description="Enable convergence study mode")
    variable: Literal["rho"] = Field("rho", description="Field used for error computation")
    norm: Literal["linf"] = Field("linf", description="Error norm (currently only linf)")
    reference: Literal["finest", "analytical_sod"] = Field(
        "finest", description="Reference solution type"
    )
    n_cases: int = Field(5, ge=2, description="Number of grid resolutions")
    n_min: int = Field(50, ge=3, description="Minimum number of cells")
    n_max: int = Field(400, ge=4, description="Maximum number of cells")
    spacing: Literal["log10", "linear"] = Field(
        "log10", description="Spacing for resolution sweep"
    )
    euler_methods: list[Literal["fv", "dg_p1", "dg_p2", "dg_p3"]] = Field(
        default_factory=lambda: ["fv", "dg_p1", "dg_p2", "dg_p3"],
        description="Euler methods to compare"
    )
    save_plot: bool = Field(True, description="Save log-log convergence plot")
    save_table: bool = Field(True, description="Save TSV table with errors")

    @model_validator(mode="after")
    def validate_bounds(self) -> "ConvergenceConfig":
        if self.n_max <= self.n_min:
            raise ValueError("convergence.n_max must be > convergence.n_min")
        if len(self.euler_methods) == 0:
            raise ValueError("convergence.euler_methods cannot be empty")
        return self


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    """Complete simulation configuration."""
    domain: DomainConfig
    time: TimeConfig
    # Physics mode (default: advection_only for backward compatibility)
    physics: Optional[PhysicsConfig] = Field(
        None, description="Physics configuration (mode, material, etc.)"
    )
    # Advection velocity (required for advection_only, not used in euler mode)
    velocity: Optional[VelocityConfig] = Field(
        None, description="Advection velocity (constant or expression). Required for advection_only mode."
    )
    # Fields for advection (required for advection_only, not used in euler mode)
    fields: Optional[list[FieldConfig]] = Field(
        None, description="Fields to advect. Required for advection_only mode."
    )
    solver: Optional[SolverConfig] = Field(
        None, description="Solver config (for advection_only mode)"
    )
    timestepper: TimeStepperConfig = Field(default_factory=TimeStepperConfig)
    sharpening: Optional[SharpeningConfig] = Field(None)
    surface_tension: Optional[SurfaceTensionConfig] = Field(None)
    convergence: Optional[ConvergenceConfig] = Field(None)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def physics_mode(self) -> str:
        """Get the physics mode (advection_only or euler)."""
        if self.physics is not None:
            return self.physics.mode
        return "advection_only"

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "SimulationConfig":
        """Validate required fields based on physics mode."""
        mode = self.physics_mode

        if mode == "advection_only":
            # Advection mode requires velocity, fields, and solver
            if self.velocity is None:
                raise ValueError(
                    "advection_only mode requires 'velocity' configuration"
                )
            if self.fields is None or len(self.fields) == 0:
                raise ValueError(
                    "advection_only mode requires at least one field in 'fields'"
                )
            if self.solver is None:
                raise ValueError(
                    "advection_only mode requires 'solver' configuration"
                )
        elif mode == "euler":
            # Euler mode: physics config is already validated
            if self.domain.ndim == 2:
                if self.physics is None:
                    raise ValueError("Euler mode requires 'physics' configuration")
                if self.physics.euler_spatial_discretization != "fv":
                    raise ValueError("2D Euler mode currently supports FV only.")
                if not self.physics.is_two_phase or self.physics.two_phase_model != "5eq":
                    raise ValueError("2D Euler mode currently supports only two-phase 5eq model.")
                if self.physics.euler_initial_conditions is None or self.physics.euler_initial_conditions.type != "rti":
                    raise ValueError("2D Euler mode currently supports only RTI initial conditions.")
            if self.convergence and self.convergence.enabled and self.domain.ndim != 1:
                raise ValueError("convergence mode currently supports only 1D")
        return self

    @model_validator(mode="after")
    def validate_velocity_dimension(self) -> "SimulationConfig":
        """Validate velocity dimension matches domain dimension."""
        if self.velocity is None:
            return self  # No velocity in euler mode

        ndim = self.domain.ndim
        vel = self.velocity
        if isinstance(vel, ConstantVelocityConfig):
            if len(vel.value) != ndim:
                raise ValueError(
                    f"velocity.value has {len(vel.value)} components but domain is {ndim}D "
                    f"(expected {ndim})"
                )
        elif isinstance(vel, ExpressionVelocityConfig):
            # For 2D, require both u and v expressions
            if ndim == 2 and vel.v is None:
                raise ValueError(
                    "2D domain requires velocity.v expression (velocity.u and velocity.v)"
                )
            # For 1D, v should be None or not used
            if ndim == 1 and vel.v is not None:
                raise ValueError(
                    "1D domain should not have velocity.v expression"
                )
        return self

    @model_validator(mode="after")
    def validate_cfl_warning(self) -> "SimulationConfig":
        """Warn if CFL > 1 (only for constant velocity in advection mode)."""
        if self.velocity is None:
            return self  # No velocity in euler mode

        vel = self.velocity
        if not isinstance(vel, ConstantVelocityConfig):
            # For expression velocity, skip CFL warning (velocity varies spatially)
            return self

        dt = self.time.dt
        dx = self.domain.dx
        u = abs(vel.value[0])
        cfl_x = u * dt / dx
        cfl = cfl_x

        if self.domain.ndim == 2:
            dy = self.domain.dy
            v = abs(vel.value[1])
            cfl_y = v * dt / dy if dy > 0 else 0.0
            cfl = max(cfl_x, cfl_y)

        if cfl > 1.0:
            import warnings
            warnings.warn(
                f"CFL = {cfl:.3f} > 1, simulation may be unstable. "
                f"Consider reducing dt or increasing n_points."
            )
        return self

    @model_validator(mode="after")
    def validate_sharpening_eps(self) -> "SimulationConfig":
        """Validate sharpening eps_target is set if enabled."""
        if self.sharpening and self.sharpening.enabled:
            if self.sharpening.eps_target is None:
                raise ValueError(
                    "sharpening.eps_target is required when sharpening is enabled"
                )
        return self


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> SimulationConfig:
    """
    Load and validate a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    SimulationConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    pydantic.ValidationError
        If the configuration is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return SimulationConfig.model_validate(raw)

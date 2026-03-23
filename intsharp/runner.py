"""
Main simulation runner (1D and 2D).

Supports two physics modes:
- advection_only: Scalar advection with optional sharpening
- euler: Compressible Euler equations with AUSM+UP flux
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .boundary import apply_bc, apply_bc_2d
from .config import (
    SimulationConfig,
    load_config,
    ConstantVelocityConfig,
    ExpressionVelocityConfig,
)
from .domain import Domain1D, Domain2D, create_domain
from .fields import (
    Field,
    create_fields,
    evaluate_velocity_expression_1d,
    evaluate_velocity_expression_2d,
)
from .registry import get_sharpening, get_solver
from .surface_tension import compute_surface_tension_diagnostics_2d
from .solvers.euler_1d import (
    EulerState1D,
    euler_step_1d,
    create_initial_state_riemann_1d,
    check_cfl_euler_1d,
    TwoPhaseEulerState1D,
    euler_step_two_phase_1d,
    create_initial_state_riemann_two_phase_1d,
    check_cfl_euler_two_phase_1d,
)
from .solvers.euler_5eq_1d import (
    FiveEqState1D,
    euler_step_5eq_1d,
    create_initial_state_riemann_5eq_1d,
    check_cfl_5eq_1d,
)
from .solvers.euler_5eq_2d import (
    FiveEqState2D,
    create_initial_state_rti_5eq_2d,
    euler_step_5eq_2d,
    check_cfl_5eq_2d,
)
from .solvers.euler_dg_1d import (
    DGEulerState1D,
    create_initial_state_riemann_dg_1d,
    dg_step_1d,
)
from .eos import total_energy_from_primitives, mixture_pressure_from_conservatives
from .exact_riemann_1d import exact_sod_1d

# Import submodules to trigger registration of solvers/monitors/sharpening/timesteppers
from . import solvers  # noqa: F401
from . import monitors  # noqa: F401
from . import sharpening as sharpening_module  # noqa: F401
from . import timesteppers as timesteppers_module  # noqa: F401

if TYPE_CHECKING:
    from .monitors.base import Monitor


def create_monitors(
    config: SimulationConfig,
    output_dir: Path,
) -> list["Monitor"]:
    """
    Create monitor instances from configuration.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    output_dir : Path
        Base output directory.

    Returns
    -------
    list[Monitor]
        List of monitor instances.
    """
    from .registry import get_monitor

    monitor_list = []

    for mon_cfg in config.output.monitors:
        monitor_cls = get_monitor(mon_cfg.type)

        # Build kwargs for monitor
        mon_output_dir = output_dir
        if mon_cfg.type == "txt":
            mon_output_dir = output_dir / "txt"
        kwargs = {
            "output_dir": mon_output_dir,
            "every_n_steps": mon_cfg.every_n_steps,
            "at_times": mon_cfg.at_times,
        }

        # Add type-specific kwargs
        if mon_cfg.type == "console":
            kwargs["total_steps"] = config.time.n_steps
        elif mon_cfg.type in ("png", "pdf", "svg"):
            kwargs["field"] = mon_cfg.field
            kwargs["show_colorbar"] = mon_cfg.show_colorbar
            kwargs["show_annotations"] = mon_cfg.show_annotations
        elif mon_cfg.type in ("gif", "mp4"):
            # Unified gif/mp4: single-field or compare mode
            kwargs["field"] = mon_cfg.field
            kwargs["style"] = mon_cfg.style
            kwargs["contour_levels"] = mon_cfg.contour_levels
            kwargs["colormap"] = mon_cfg.colormap
            kwargs["contour_overlay_color"] = mon_cfg.contour_overlay_color
            kwargs["contour_color"] = mon_cfg.contour_color
            kwargs["background_color"] = mon_cfg.background_color
            kwargs["show_colorbar"] = mon_cfg.show_colorbar
            kwargs["show_annotations"] = mon_cfg.show_annotations
            kwargs["vmin"] = mon_cfg.vmin
            kwargs["vmax"] = mon_cfg.vmax
            kwargs["output_format"] = mon_cfg.type  # "gif" or "mp4"
            kwargs["quiver_overlay_x"] = mon_cfg.quiver_overlay_x
            kwargs["quiver_overlay_y"] = mon_cfg.quiver_overlay_y
            kwargs["quiver_skip"] = mon_cfg.quiver_skip
            if mon_cfg.compare_fields:
                kwargs["compare_fields"] = [
                    {
                        "field": cf.field,
                        "contour_levels": cf.contour_levels,
                        "color": cf.color,
                        "linestyle": cf.linestyle,
                    }
                    for cf in mon_cfg.compare_fields
                ]
        elif mon_cfg.type in ("txt", "curve"):
            kwargs["field"] = mon_cfg.field
            kwargs["fields"] = mon_cfg.fields
        elif mon_cfg.type == "hdf5":
            kwargs["fields"] = mon_cfg.fields

        monitor_list.append(monitor_cls(**kwargs))

    return monitor_list


def run_advection_simulation(
    config: SimulationConfig,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Run advection-only simulation (original mode).

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration.
    config_dir : Path or None
        Directory of the config file for resolving relative image paths.

    Returns
    -------
    dict[str, Field]
        Final field states.
    """
    # Create domain
    domain = create_domain(config.domain)
    ndim = domain.ndim

    # Create fields
    fields = create_fields(config.fields, domain, config_dir)

    # Get solver based on dimension
    if ndim == 1:
        solver_fn = get_solver(config.solver.type)
    else:
        # For 2D, use the 2D version of the solver
        solver_type_2d = config.solver.type + "_2d"
        try:
            solver_fn = get_solver(solver_type_2d)
        except KeyError:
            # Fall back to the specified solver if no 2D version exists
            solver_fn = get_solver(config.solver.type)

    # Check if sharpening is needed at all (global enabled OR any per-field override)
    needs_sharpening = False
    if config.sharpening:
        if config.sharpening.enabled:
            needs_sharpening = True
        else:
            for field_cfg in config.fields:
                if field_cfg.sharpening is True:
                    needs_sharpening = True
                    break

    # Cache for resolved sharpening functions (method_name -> callable)
    _sharpening_fn_cache: dict[str, object] = {}

    def _resolve_sharpening_fn(method_name: str):
        """Resolve sharpening function by name, with 2D suffix fallback."""
        if method_name in _sharpening_fn_cache:
            return _sharpening_fn_cache[method_name]
        if ndim == 1:
            fn = get_sharpening(method_name)
        else:
            try:
                fn = get_sharpening(method_name + "_2d")
            except KeyError:
                fn = get_sharpening(method_name)
        _sharpening_fn_cache[method_name] = fn
        return fn

    # Create output directory
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create monitors
    monitor_list = create_monitors(config, output_dir)

    # Extract parameters
    dt = config.time.dt
    n_steps = config.time.n_steps
    dx = domain.dx

    if ndim == 1:
        dy = 0.0
    else:
        dy = domain.dy  # type: ignore

    # Velocity configuration: constant or expression
    vel_config = config.velocity
    is_constant_velocity = isinstance(vel_config, ConstantVelocityConfig)

    # For constant velocity, extract scalar values once
    if is_constant_velocity:
        if ndim == 1:
            const_velocity = vel_config.value[0]
        else:
            const_velocity = (vel_config.value[0], vel_config.value[1])

    # Track primary field names (only these are advected)
    primary_field_names = set(fields.keys())

    # Initialize monitors
    for monitor in monitor_list:
        monitor.on_start(fields, domain)

    # Surface tension diagnostics at t=0 (so monitors can output curvature etc. from start)
    t = 0.0
    if ndim == 2 and config.surface_tension and config.surface_tension.enabled:
        source_name = config.surface_tension.source_field
        if source_name in fields:
            source = fields[source_name]
            sigma = config.surface_tension.sigma
            smoothing_sigma = config.surface_tension.smoothing_sigma
            band_min = config.surface_tension.interface_band_alpha_min
            band_max = config.surface_tension.interface_band_alpha_max
            st_diags = compute_surface_tension_diagnostics_2d(
                source.values, sigma, dx, dy, source.bc,
                smoothing_sigma=smoothing_sigma,
                interface_band_alpha_min=band_min,
                interface_band_alpha_max=band_max,
            )
            for diag_name, diag_values in st_diags.items():
                fields[diag_name] = Field(diag_name, diag_values, source.bc, None)

    # Output initial conditions (step 0)
    for monitor in monitor_list:
        monitor.on_step(0, t, fields, domain)

    # Time loop
    for step in range(1, n_steps + 1):
        # Compute velocity for this step
        if is_constant_velocity:
            velocity = const_velocity
        else:
            # Expression velocity: evaluate at current time t
            if ndim == 1:
                velocity = evaluate_velocity_expression_1d(
                    vel_config.u,
                    domain.x,
                    t,
                )
            else:
                velocity = evaluate_velocity_expression_2d(
                    vel_config.u,
                    vel_config.v,  # type: ignore (validated to exist for 2D)
                    domain.X,
                    domain.Y,
                    t,
                )

        # Advect each primary field (skip derived diagnostic fields)
        for name in primary_field_names:
            field = fields[name]
            if ndim == 1:
                # 1D advection
                new_values = solver_fn(
                    field.values,
                    velocity,
                    dx,
                    dt,
                    field.bc,
                )
                # Apply boundary conditions
                new_values = apply_bc(new_values, field.bc, dx)
            else:
                # 2D advection
                new_values = solver_fn(
                    field.values,
                    velocity,
                    dx,
                    dy,
                    dt,
                    field.bc,
                )
                # Apply boundary conditions
                new_values = apply_bc_2d(new_values, field.bc, dx, dy)

            field.values = new_values

        # Sharpening post-step (only primary fields)
        if config.sharpening is not None and needs_sharpening:
            method_params = config.sharpening.method_params or {}
            for name in primary_field_names:
                field = fields[name]
                if field.sharpening_enabled is True:
                    do_sharp = True
                elif field.sharpening_enabled is False:
                    do_sharp = False
                else:
                    do_sharp = config.sharpening.enabled

                if do_sharp:
                    method_name = field.sharpening_method or config.sharpening.method
                    sharp_fn = _resolve_sharpening_fn(method_name)
                    if ndim == 1:
                        field.values = sharp_fn(
                            field.values,
                            dx,
                            dt,
                            config.sharpening.eps_target,
                            config.sharpening.strength,
                            field.bc,
                            **method_params,
                        )
                    else:
                        field.values = sharp_fn(
                            field.values,
                            dx,
                            dy,
                            dt,
                            config.sharpening.eps_target,
                            config.sharpening.strength,
                            field.bc,
                            **method_params,
                        )

        # Surface tension diagnostics (2D only)
        if ndim == 2 and config.surface_tension and config.surface_tension.enabled:
            source_name = config.surface_tension.source_field
            if source_name in fields:
                source = fields[source_name]
                sigma = config.surface_tension.sigma
                
                # Compute all surface tension diagnostics
                smoothing_sigma = config.surface_tension.smoothing_sigma
                band_min = config.surface_tension.interface_band_alpha_min
                band_max = config.surface_tension.interface_band_alpha_max
                st_diags = compute_surface_tension_diagnostics_2d(
                    source.values, sigma, dx, dy, source.bc,
                    smoothing_sigma=smoothing_sigma,
                    interface_band_alpha_min=band_min,
                    interface_band_alpha_max=band_max,
                )
                
                # Add as derived fields (use source BC, no sharpening)
                for diag_name, diag_values in st_diags.items():
                    fields[diag_name] = Field(diag_name, diag_values, source.bc, None)

        # Update time
        t += dt

        # Call monitors
        for monitor in monitor_list:
            monitor.on_step(step, t, fields, domain)

    # Finalize monitors
    for monitor in monitor_list:
        monitor.on_end(fields, domain)

    return fields


def run_euler_simulation(
    config: SimulationConfig,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Run compressible Euler simulation (single-phase or two-phase).

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration.
    config_dir : Path or None
        Directory of the config file (unused for Euler mode).

    Returns
    -------
    dict[str, Field]
        Final field states (rho, u, p, E, and alpha for two-phase).
    """
    from .boundary import BoundaryCondition

    # Create domain
    domain = create_domain(config.domain)

    # Extract physics config
    physics = config.physics
    euler_ic = physics.euler_initial_conditions
    euler_bc = physics.euler_bc
    euler_bc_x = physics.euler_bc_x
    euler_bc_y = physics.euler_bc_y
    flux_calculator = physics.flux_calculator
    euler_discretization = physics.euler_spatial_discretization

    # Determine if single-phase or two-phase
    is_two_phase = physics.is_two_phase

    if domain.ndim == 2:
        if not is_two_phase or physics.two_phase_model != "5eq":
            raise ValueError("2D Euler currently supports two-phase 5eq only")
        if euler_ic.type != "rti":
            raise ValueError("2D Euler currently supports RTI initial conditions only")
        if physics.phase1.rho_ref is None or physics.phase2.rho_ref is None:
            raise ValueError("2D RTI setup requires phase1.rho_ref and phase2.rho_ref")

        gamma1 = physics.phase1.gamma
        p_inf1 = physics.phase1.p_infinity
        gamma2 = physics.phase2.gamma
        p_inf2 = physics.phase2.p_infinity
        use_muscl = physics.use_muscl
        gravity = physics.gravity

        state = create_initial_state_rti_5eq_2d(
            X=domain.X,  # type: ignore[attr-defined]
            Y=domain.Y,  # type: ignore[attr-defined]
            rho1=physics.phase1.rho_ref,
            rho2=physics.phase2.rho_ref,
            gamma1=gamma1,
            gamma2=gamma2,
            p_inf1=p_inf1,
            p_inf2=p_inf2,
            interface_y0=euler_ic.interface_y0,
            alpha1_top=euler_ic.alpha1_top,
            alpha1_bottom=euler_ic.alpha1_bottom,
            perturbation_amplitude=euler_ic.perturbation_amplitude,
            perturbation_mode_x=euler_ic.perturbation_mode_x,
            interface_thickness=euler_ic.interface_thickness,
            p0=euler_ic.p0,
            u0=euler_ic.u0,
            v0=euler_ic.v0,
            gravity_y=gravity.gy,
            v_perturbation_amplitude=euler_ic.v_perturbation_amplitude,
            v_perturbation_mode_x=euler_ic.v_perturbation_mode_x,
            v_perturbation_scale_with_sound_speed=euler_ic.v_perturbation_scale_with_sound_speed,
        )

        dummy_bc = BoundaryCondition("neumann", gradient_left=0.0, gradient_right=0.0)

        def state_to_fields(s: FiveEqState2D) -> dict[str, Field]:
            p = s.p
            rho = s.rho
            u = s.u
            v = s.v
            gamma_eff = s.alpha1 * gamma1 + (1.0 - s.alpha1) * gamma2
            p_inf_eff = s.alpha1 * p_inf1 + (1.0 - s.alpha1) * p_inf2
            e = (p + gamma_eff * p_inf_eff) / ((gamma_eff - 1.0) * rho + 1e-30)
            return {
                "rho": Field("rho", rho.copy(), dummy_bc, None),
                "u": Field("u", u.copy(), dummy_bc, None),
                "v": Field("v", v.copy(), dummy_bc, None),
                "p": Field("p", p.copy(), dummy_bc, None),
                "E": Field("E", s.E.copy(), dummy_bc, None),
                "e_int": Field("e_int", e.copy(), dummy_bc, None),
                "alpha": Field("alpha", s.alpha1.copy(), dummy_bc, None),
                "alpha1_rho1": Field("alpha1_rho1", s.alpha1_rho1.copy(), dummy_bc, None),
                "alpha2_rho2": Field("alpha2_rho2", s.alpha2_rho2.copy(), dummy_bc, None),
                "rho_u": Field("rho_u", s.rho_u.copy(), dummy_bc, None),
                "rho_v": Field("rho_v", s.rho_v.copy(), dummy_bc, None),
            }

        def step_fn(s, dx, dy, dt):
            return euler_step_5eq_2d(
                s,
                dx=dx,
                dy=dy,
                dt=dt,
                bc_x=euler_bc_x,
                bc_y=euler_bc_y,
                use_muscl=use_muscl,
                flux_calculator=flux_calculator,
                gravity_x=gravity.gx,
                gravity_y=gravity.gy,
                gravity_enabled=gravity.enabled,
            )

        def cfl_fn(s, dx, dy, dt):
            return check_cfl_5eq_2d(s, dx, dy, dt)

    elif is_two_phase:
        # Two-phase mode
        gamma1 = physics.phase1.gamma
        p_inf1 = physics.phase1.p_infinity
        gamma2 = physics.phase2.gamma
        p_inf2 = physics.phase2.p_infinity
        two_phase_model = physics.two_phase_model
        use_muscl = physics.use_muscl

        # Get alpha from IC (default to 1.0 on left, 0.0 on right if not specified)
        if euler_ic.type != "riemann":
            raise ValueError(f"Two-phase Euler only supports 'riemann' IC, got: {euler_ic.type}")
        
        alpha_L = euler_ic.left.alpha if euler_ic.left.alpha is not None else 1.0
        alpha_R = euler_ic.right.alpha if euler_ic.right.alpha is not None else 0.0

        dummy_bc = BoundaryCondition("neumann", gradient_left=0.0, gradient_right=0.0)

        if two_phase_model == "5eq":
            # 5-equation model
            # For 5-eq, we need phase densities, not mixture density
            # Left: if alpha=1, pure phase 1, so rho1=rho, rho2 is ambient
            # Right: if alpha=0, pure phase 2, so rho2=rho, rho1 is ambient
            rho1_L = euler_ic.left.rho if alpha_L > 0.5 else euler_ic.left.rho
            rho2_L = euler_ic.left.rho if alpha_L < 0.5 else euler_ic.left.rho
            rho1_R = euler_ic.right.rho if alpha_R > 0.5 else euler_ic.right.rho
            rho2_R = euler_ic.right.rho if alpha_R < 0.5 else euler_ic.right.rho
            
            # Actually, for water-air: left is water (alpha=1), right is air (alpha=0)
            # rho_L = 1000 is water density, rho_R = 10 is air density
            # So: rho1_L = 1000 (water), rho2_L = any (doesn't matter, alpha2=0)
            #     rho1_R = any (doesn't matter, alpha1=0), rho2_R = 10 (air)
            # For generality, use rho from IC as the dominant phase density
            rho1_L = euler_ic.left.rho   # Phase 1 density on left
            rho2_L = euler_ic.right.rho  # Use air density as phase 2 on left (trace)
            rho1_R = euler_ic.left.rho   # Use water density as phase 1 on right (trace)
            rho2_R = euler_ic.right.rho  # Phase 2 density on right

            state = create_initial_state_riemann_5eq_1d(
                x=domain.x,
                x_discontinuity=euler_ic.x_discontinuity,
                rho1_L=rho1_L,
                rho2_L=rho2_L,
                u_L=euler_ic.left.u,
                p_L=euler_ic.left.p,
                alpha1_L=alpha_L,
                rho1_R=rho1_R,
                rho2_R=rho2_R,
                u_R=euler_ic.right.u,
                p_R=euler_ic.right.p,
                alpha1_R=alpha_R,
                gamma1=gamma1,
                gamma2=gamma2,
                p_inf1=p_inf1,
                p_inf2=p_inf2,
            )

            def state_to_fields(s: FiveEqState1D) -> dict[str, Field]:
                gamma_eff = s.alpha1 * gamma1 + s.alpha2 * gamma2
                p_inf_eff = s.alpha1 * p_inf1 + s.alpha2 * p_inf2
                e = (s.p + gamma_eff * p_inf_eff) / ((gamma_eff - 1.0) * s.rho + 1e-30)
                return {
                    "rho": Field("rho", s.rho.copy(), dummy_bc, None),
                    "u": Field("u", s.u.copy(), dummy_bc, None),
                    "p": Field("p", s.p.copy(), dummy_bc, None),
                    "E": Field("E", s.E.copy(), dummy_bc, None),
                    "e_int": Field("e_int", e.copy(), dummy_bc, None),
                    "alpha": Field("alpha", s.alpha1.copy(), dummy_bc, None),
                    "alpha1_rho1": Field("alpha1_rho1", s.alpha1_rho1.copy(), dummy_bc, None),
                    "alpha2_rho2": Field("alpha2_rho2", s.alpha2_rho2.copy(), dummy_bc, None),
                    "rho_u": Field("rho_u", s.rho_u.copy(), dummy_bc, None),
                }

            def step_fn(s, dx, dt, bc):
                return euler_step_5eq_1d(
                    s, dx, dt, bc, use_muscl=use_muscl, flux_calculator=flux_calculator
                )

            def cfl_fn(s, dx, dt):
                return check_cfl_5eq_1d(s, dx, dt)

        else:
            # Mixture model (original two-phase)
            state = create_initial_state_riemann_two_phase_1d(
                x=domain.x,
                x_discontinuity=euler_ic.x_discontinuity,
                rho_L=euler_ic.left.rho,
                u_L=euler_ic.left.u,
                p_L=euler_ic.left.p,
                alpha_L=alpha_L,
                rho_R=euler_ic.right.rho,
                u_R=euler_ic.right.u,
                p_R=euler_ic.right.p,
                alpha_R=alpha_R,
                gamma1=gamma1,
                gamma2=gamma2,
                p_inf1=p_inf1,
                p_inf2=p_inf2,
            )

            def state_to_fields(s: TwoPhaseEulerState1D) -> dict[str, Field]:
                gamma_eff = s.alpha * gamma1 + (1.0 - s.alpha) * gamma2
                p_inf_eff = s.alpha * p_inf1 + (1.0 - s.alpha) * p_inf2
                e = (s.p + gamma_eff * p_inf_eff) / ((gamma_eff - 1.0) * s.rho + 1e-30)
                return {
                    "rho": Field("rho", s.rho.copy(), dummy_bc, None),
                    "u": Field("u", s.u.copy(), dummy_bc, None),
                    "p": Field("p", s.p.copy(), dummy_bc, None),
                    "E": Field("E", s.E.copy(), dummy_bc, None),
                    "e_int": Field("e_int", e.copy(), dummy_bc, None),
                    "alpha": Field("alpha", s.alpha.copy(), dummy_bc, None),
                    "rho_u": Field("rho_u", s.rho_u.copy(), dummy_bc, None),
                }

            def step_fn(s, dx, dt, bc):
                return euler_step_two_phase_1d(
                    s, dx, dt, bc, use_muscl=use_muscl, flux_calculator=flux_calculator
                )

            def cfl_fn(s, dx, dt):
                return check_cfl_euler_two_phase_1d(s, dx, dt)

    else:
        # Single-phase mode
        material = physics.material
        gamma = material.gamma
        p_inf = material.p_infinity

        # Create initial state
        if euler_ic.type == "riemann":
            if euler_discretization == "dg":
                state = create_initial_state_riemann_dg_1d(
                    x=domain.x,
                    x_discontinuity=euler_ic.x_discontinuity,
                    rho_L=euler_ic.left.rho,
                    u_L=euler_ic.left.u,
                    p_L=euler_ic.left.p,
                    rho_R=euler_ic.right.rho,
                    u_R=euler_ic.right.u,
                    p_R=euler_ic.right.p,
                    gamma=gamma,
                    p_inf=p_inf,
                    order=physics.dg_order,
                )
            else:
                state = create_initial_state_riemann_1d(
                    x=domain.x,
                    x_discontinuity=euler_ic.x_discontinuity,
                    rho_L=euler_ic.left.rho,
                    u_L=euler_ic.left.u,
                    p_L=euler_ic.left.p,
                    rho_R=euler_ic.right.rho,
                    u_R=euler_ic.right.u,
                    p_R=euler_ic.right.p,
                    gamma=gamma,
                    p_inf=p_inf,
                )
        elif euler_ic.type == "uniform":
            n = len(domain.x)
            rho = np.full(n, euler_ic.rho)
            u = np.full(n, euler_ic.u)
            p = np.full(n, euler_ic.p)
            if euler_discretization == "dg":
                rho_u = rho * u
                E = total_energy_from_primitives(rho, u, p, gamma, p_inf)
                Ubar = np.column_stack([rho, rho_u, E]).astype(np.float64)
                n_nodes = physics.dg_order + 1
                U = np.repeat(Ubar[:, None, :], n_nodes, axis=1)
                state = DGEulerState1D(U=U, order=physics.dg_order, gamma=gamma, p_inf=p_inf)
            else:
                rho_u = rho * u
                E = total_energy_from_primitives(rho, u, p, gamma, p_inf)
                state = EulerState1D(rho=rho, rho_u=rho_u, E=E, gamma=gamma, p_inf=p_inf)
        else:
            raise ValueError(f"Unknown Euler IC type: {euler_ic.type}")

        # State to fields converter for single-phase
        dummy_bc = BoundaryCondition("neumann", gradient_left=0.0, gradient_right=0.0)

        if euler_discretization == "dg":
            def state_to_fields(s: DGEulerState1D) -> dict[str, Field]:
                p = s.p
                rho = s.rho
                e = (p + gamma * p_inf) / ((gamma - 1.0) * rho)
                return {
                    "rho": Field("rho", rho.copy(), dummy_bc, None),
                    "u": Field("u", s.u.copy(), dummy_bc, None),
                    "p": Field("p", p.copy(), dummy_bc, None),
                    "E": Field("E", s.E.copy(), dummy_bc, None),
                    "e_int": Field("e_int", e.copy(), dummy_bc, None),
                    "rho_u": Field("rho_u", s.rho_u.copy(), dummy_bc, None),
                }
        else:
            def state_to_fields(s: EulerState1D) -> dict[str, Field]:
                e = (s.p + gamma * p_inf) / ((gamma - 1.0) * s.rho)
                return {
                    "rho": Field("rho", s.rho.copy(), dummy_bc, None),
                    "u": Field("u", s.u.copy(), dummy_bc, None),
                    "p": Field("p", s.p.copy(), dummy_bc, None),
                    "E": Field("E", s.E.copy(), dummy_bc, None),
                    "e_int": Field("e_int", e.copy(), dummy_bc, None),
                    "rho_u": Field("rho_u", s.rho_u.copy(), dummy_bc, None),
                }

        # Time stepping function
        use_muscl = physics.use_muscl
        if euler_discretization == "dg":
            dg_use_limiter = physics.dg_use_limiter
            dg_use_positivity = physics.dg_use_positivity

            def step_fn(s, dx, dt, bc):
                return dg_step_1d(
                    s, dx, dt, bc_type=bc, use_limiter=dg_use_limiter, use_positivity=dg_use_positivity
                )
        else:
            def step_fn(s, dx, dt, bc):
                return euler_step_1d(
                    s, dx, dt, bc, use_muscl=use_muscl, flux_calculator=flux_calculator
                )

        # CFL check function
        def cfl_fn(s, dx, dt):
            return check_cfl_euler_1d(s, dx, dt)

    # Create output directory
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create monitors
    monitor_list = create_monitors(config, output_dir)

    # Extract time parameters
    dt = config.time.dt
    n_steps = config.time.n_steps
    dx = domain.dx
    dy = getattr(domain, "dy", 0.0)

    # Check CFL at start
    if domain.ndim == 1:
        cfl_fn(state, dx, dt)
    else:
        cfl_fn(state, dx, dy, dt)

    # Initialize monitors
    fields = state_to_fields(state)
    for monitor in monitor_list:
        monitor.on_start(fields, domain)

    # Output initial conditions (step 0)
    t = 0.0
    for monitor in monitor_list:
        monitor.on_step(0, t, fields, domain)

    # Time loop
    for step in range(1, n_steps + 1):
        # Euler step
        if domain.ndim == 1:
            state = step_fn(state, dx, dt, euler_bc)
        else:
            state = step_fn(state, dx, dy, dt)
        t += dt

        # Convert to fields for monitors
        fields = state_to_fields(state)

        # Call monitors
        for monitor in monitor_list:
            monitor.on_step(step, t, fields, domain)

    # Finalize monitors
    for monitor in monitor_list:
        monitor.on_end(fields, domain)

    return fields


def run_simulation(
    config: SimulationConfig,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Run the simulation.

    Dispatches to the appropriate runner based on physics mode.

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration.
    config_dir : Path or None
        Directory of the config file for resolving relative image paths.

    Returns
    -------
    dict[str, Field]
        Final field states.
    """
    if config.convergence and config.convergence.enabled:
        return run_convergence_study(config, config_dir)

    mode = config.physics_mode

    if mode == "euler":
        return run_euler_simulation(config, config_dir)
    else:
        return run_advection_simulation(config, config_dir)


def run_from_file(config_path: str | Path) -> dict[str, Field]:
    """
    Load configuration from file and run simulation.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    dict[str, Field]
        Final field states.
    """
    config = load_config(config_path)
    return run_simulation(config)


def _build_resolution_sweep(n_min: int, n_max: int, n_cases: int, spacing: str) -> list[int]:
    """Build monotonically increasing integer grid counts."""
    if spacing == "log10":
        vals = np.geomspace(n_min, n_max, n_cases)
    else:
        vals = np.linspace(n_min, n_max, n_cases)
    vals = np.rint(vals).astype(int)
    vals[0] = n_min
    vals[-1] = n_max

    # Enforce strict monotonicity
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + 1
    for i in range(len(vals) - 2, -1, -1):
        if vals[i] >= vals[i + 1]:
            vals[i] = vals[i + 1] - 1

    vals = np.clip(vals, n_min, n_max)
    vals = np.unique(vals)
    if len(vals) < n_cases:
        pool = [v for v in range(n_min, n_max + 1) if v not in set(vals.tolist())]
        while len(vals) < n_cases and pool:
            target = np.linspace(n_min, n_max, n_cases)[len(vals)]
            pick = min(pool, key=lambda p: abs(p - target))
            vals = np.unique(np.append(vals, pick))
            pool.remove(pick)
    vals = np.sort(vals)
    return [int(v) for v in vals]


def run_convergence_study(
    config: SimulationConfig,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Run configurable spatial convergence study for 1D Euler.

    Computes Linf error in selected variable against each method's finest-grid
    solution, fits observed order, and saves table/plot in output directory.
    """
    conv = config.convergence
    if conv is None or not conv.enabled:
        raise ValueError("run_convergence_study called without enabled convergence config")
    if config.physics_mode != "euler":
        raise ValueError("convergence mode currently supports Euler physics only")
    if config.physics is None:
        raise ValueError("Euler convergence mode requires physics configuration")
    if config.domain.ndim != 1:
        raise ValueError("convergence mode currently supports only 1D domains")
    if config.physics.is_two_phase and any(m.startswith("dg_") for m in conv.euler_methods):
        raise ValueError("DG convergence methods are currently single-phase only")

    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_values = _build_resolution_sweep(conv.n_min, conv.n_max, conv.n_cases, conv.spacing)
    methods = conv.euler_methods
    L = config.domain.x_max - config.domain.x_min

    # method -> n -> (x, rho)
    results: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {m: {} for m in methods}
    final_fields_by_method: dict[str, dict[str, Field]] = {}

    for method in methods:
        for n in n_values:
            cfg = copy.deepcopy(config)
            if cfg.convergence:
                cfg.convergence.enabled = False
            cfg.domain.n_points = n
            cfg.domain.n_points_x = n
            cfg.output.monitors = []
            cfg.output.directory = str(output_dir / "_tmp_convergence_runs")

            if method == "fv":
                cfg.physics.euler_spatial_discretization = "fv"
            elif method == "dg_p1":
                cfg.physics.euler_spatial_discretization = "dg"
                cfg.physics.dg_order = 1
            elif method == "dg_p2":
                cfg.physics.euler_spatial_discretization = "dg"
                cfg.physics.dg_order = 2
            elif method == "dg_p3":
                cfg.physics.euler_spatial_discretization = "dg"
                cfg.physics.dg_order = 3
            else:
                raise ValueError(f"Unsupported convergence method: {method}")

            fields = run_euler_simulation(cfg, config_dir=config_dir)
            x = np.linspace(cfg.domain.x_min, cfg.domain.x_max, n)
            rho = fields["rho"].values.copy()
            results[method][n] = (x, rho)
            final_fields_by_method[method] = fields

    rows: list[tuple[str, int, float, float]] = []
    observed_orders: dict[str, float] = {}

    for method in methods:
        if conv.reference == "analytical_sod":
            ic = config.physics.euler_initial_conditions
            if (
                config.physics is None
                or config.physics.material is None
                or config.physics.material.p_infinity != 0.0
                or ic is None
                or ic.type != "riemann"
            ):
                raise ValueError(
                    "analytical_sod reference requires single-phase ideal-gas Riemann setup"
                )
            gamma = config.physics.material.gamma
            x0 = ic.x_discontinuity
            left = ic.left
            right = ic.right
            t_final = config.time.dt * config.time.n_steps
        else:
            n_ref = max(n_values)
            x_ref, rho_ref = results[method][n_ref]

        dx_vals = []
        err_vals = []
        for n in n_values:
            x_k, rho_k = results[method][n]
            if conv.reference == "analytical_sod":
                rho_exact, _, _ = exact_sod_1d(
                    x_k,
                    t=t_final,
                    x_discontinuity=x0,
                    rho_L=left.rho,
                    u_L=left.u,
                    p_L=left.p,
                    rho_R=right.rho,
                    u_R=right.u,
                    p_R=right.p,
                    gamma=gamma,
                )
                err = float(np.max(np.abs(rho_k - rho_exact)))
            else:
                rho_ref_on_k = np.interp(x_k, x_ref, rho_ref)
                err = float(np.max(np.abs(rho_k - rho_ref_on_k)))
            dx = L / n
            rows.append((method, n, dx, err))
            if err > 0.0:
                dx_vals.append(dx)
                err_vals.append(err)
        if len(dx_vals) >= 2:
            slope, _ = np.polyfit(np.log(dx_vals), np.log(err_vals), 1)
            observed_orders[method] = float(slope)
        else:
            observed_orders[method] = float("nan")

    if conv.save_table:
        table_path = output_dir / "convergence_rho_linf.tsv"
        with open(table_path, "w") as f:
            f.write("method\tn_points\tdx\terror_linf\n")
            for method, n, dx, err in rows:
                f.write(f"{method}\t{n}\t{dx:.16e}\t{err:.16e}\n")
        summary_path = output_dir / "convergence_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Observed order from log(error)=p*log(dx)+c (excluding zero error points)\n")
            for method in methods:
                p = observed_orders[method]
                f.write(f"{method}: order={p:.6f}\n")

        # Persist per-case profiles so reference changes can be re-postprocessed
        profiles_dir = output_dir / "convergence_profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            for n in n_values:
                x_k, rho_k = results[method][n]
                prof_path = profiles_dir / f"{method}_n{n}.tsv"
                np.savetxt(
                    prof_path,
                    np.column_stack([x_k, rho_k]),
                    delimiter="\t",
                    header="x\trho",
                    comments="",
                )

    if conv.save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5.5))
        for method in methods:
            method_rows = [(n, dx, err) for m, n, dx, err in rows if m == method and err > 0.0]
            method_rows.sort(key=lambda t: t[1], reverse=True)
            if len(method_rows) == 0:
                continue
            dx_plot = [r[1] for r in method_rows]
            err_plot = [r[2] for r in method_rows]
            p = observed_orders[method]
            label = f"{method} (order={p:.3f})" if np.isfinite(p) else method
            plt.loglog(dx_plot, err_plot, marker="o", linewidth=1.5, label=label)

        plt.xlabel("Grid spacing dx")
        if conv.reference == "analytical_sod":
            ylab = "Linf error in rho (vs analytical Sod)"
            ttl = "Spatial Convergence: Euler 1D (Analytical Sod Reference)"
        else:
            ylab = "Linf error in rho (vs finest per method)"
            ttl = "Spatial Convergence: Euler 1D"
        plt.ylabel(ylab)
        plt.title(ttl)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_rho_linf.png", dpi=170)
        plt.close()

    # Return finest-grid fields for first requested method for API compatibility
    first_method = methods[0]
    return final_fields_by_method[first_method]

"""
Polymerization Kinetics Model
=============================

Numerical simulation of polymerization kinetics using a system of
ordinary differential equations (ODEs).

The model describes:
- monomer conversion over time;
- number-average and weight-average molecular weights;
- crosslink formation;
- the influence of kinetic parameters and initiator concentration.

The stiff ODE system is solved with SciPy's BDF integrator.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Physical and model constants
# ---------------------------------------------------------------------

NA = 6.022e23          # Avogadro constant, mol^-1
KB = 1.3806e-23        # Boltzmann constant, J/K

MW_CRITICAL = 1e4
ALPHA0_M = 0.149
ALPHA0_P = 0.0194
D_ALPHA_M = 2.9e-4
D_ALPHA_P = 0.13e-4

TG_P = 378.15          # K
TG_M = 159.15          # K

EPSILON = 0.0
F_SEG = 1.0

GAMMA_P = 1.2
VM_STAR = 0.868
VP_STAR = 0.788
VF_M = 0.1751
DZETA_MP = 0.59

JC0 = 187.81
XC0 = 100.0
SIGMA = 6.9

MONOMER_INITIAL = 5.5
MONOMER_MOLAR_MASS = 100.13
EPS = 1e-20


# ---------------------------------------------------------------------
# Auxiliary physical relations
# ---------------------------------------------------------------------

def polymer_volume_fraction(conversion: float) -> float:
    """Return polymer volume fraction as a function of conversion."""
    return (1.0 + EPSILON) * conversion / (1.0 + EPSILON * conversion)


def free_volume(conversion: float, temperature: float) -> float:
    """Estimate the mixture free-volume fraction."""
    phi_p = polymer_volume_fraction(conversion)

    vf_p = ALPHA0_P + D_ALPHA_P * (temperature - TG_P)
    vf_m = ALPHA0_M + D_ALPHA_M * (temperature - TG_M)

    return vf_p * phi_p + vf_m * (1.0 - phi_p)


def hydrodynamic_radius(molecular_weight: float) -> float:
    """Estimate hydrodynamic radius from molecular weight."""
    mw_safe = max(molecular_weight, 100.0)

    intrinsic_viscosity = 6.63e-3 * mw_safe**0.73
    radius_cm = (
        3.0 * intrinsic_viscosity * mw_safe / (10.0 * np.pi * NA)
    ) ** (1.0 / 3.0)

    return max(radius_cm * 0.01, 1e-12)


def viscosity(temperature: float) -> float:
    """Return viscosity from the empirical temperature relation."""
    exponent = 453.25 * (1.0 / temperature - 1.0 / 254.92)
    return 10.0**exponent


def diffusion_coefficient(temperature: float, molecular_weight: float) -> float:
    """Calculate diffusion coefficient using the Stokes-Einstein relation."""
    radius = max(hydrodynamic_radius(molecular_weight), 1e-12)
    viscosity_pas = max(viscosity(temperature) * 1e-3, 1e-12)

    return KB * temperature / (6.0 * np.pi * viscosity_pas * radius)


def polymer_diffusion(
    diffusion_0: float,
    molecular_weight: float,
    vf: float,
    omega_m: float,
    omega_p: float,
) -> float:
    """Return the effective polymer diffusion term used in the model."""
    mw_safe = max(molecular_weight, MW_CRITICAL)
    vf_safe = max(vf, 1e-6)

    molecular_weight_term = (mw_safe / MW_CRITICAL) ** 2

    exponent = -GAMMA_P * (
        (omega_m * VM_STAR / DZETA_MP + omega_p * VP_STAR) / vf_safe
        - (VM_STAR / DZETA_MP) / VF_M
    )

    return (diffusion_0 / molecular_weight_term) * np.exp(exponent)


def critical_chain_length(conversion: float) -> float:
    """Return conversion-dependent critical chain length."""
    return 1.0 / (1.0 / JC0 + 2.0 * conversion / XC0)


def tau(conversion: float) -> float:
    """Auxiliary model parameter."""
    jc_value = critical_chain_length(conversion)
    return np.sqrt(3.0 / (2.0 * jc_value * SIGMA**2))


def termination_radius(conversion: float, y0: float) -> float:
    """Return the effective termination radius."""
    y0_safe = max(y0, 1e-12)
    tau_value = tau(conversion)

    term = (
        1000.0 * tau_value**3
        / (NA * y0_safe * np.pi**3.5)
    )

    if term <= 1.0000001:
        return 1e-9

    return max(np.sqrt(np.log(term)) / tau_value, 1e-12)


def intrinsic_termination_rate(_: float) -> float:
    """Intrinsic termination rate constant."""
    return 2.2e6


def termination_rate(
    conversion: float,
    temperature: float,
    molecular_weight: float,
    y0: float,
) -> float:
    """Calculate the effective termination rate."""
    conversion = np.clip(conversion, 0.0, 0.999)

    vf = free_volume(conversion, temperature)
    omega_m = 1.0 - conversion
    omega_p = conversion

    diffusion_0 = diffusion_coefficient(temperature, molecular_weight)
    diffusion_eff = max(
        F_SEG * polymer_diffusion(
            diffusion_0,
            molecular_weight,
            vf,
            omega_m,
            omega_p,
        ),
        1e-20,
    )

    radius = max(termination_radius(conversion, y0), 1e-12)

    return 1.0 / (
        1.0 / intrinsic_termination_rate(temperature)
        + 1.0 / (4.0 * np.pi * radius * diffusion_eff * NA * 1000.0)
    )


# ---------------------------------------------------------------------
# ODE model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ModelParameters:
    """Kinetic parameters for one simulation."""

    f: float = 0.5
    kd: float = 0.03844
    kp: float = 400.0
    k1p: float = 5.0
    initiator_0: float = 0.001
    temperature: float = 298.0


def solve_polymerization(
    params: ModelParameters,
    time_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the polymerization model on the supplied time grid.

    State vector:
        I, M, pQ1, Y0, Y1, Y2, Q0, Q1, Q2
    """

    def rhs(_: float, state: np.ndarray) -> list[float]:
        I, M, pQ1, Y0, Y1, Y2, Q0, Q1, Q2 = state

        q3 = 3.0 * Q1 * Q2 - Q1**3

        conversion = np.clip(
            (MONOMER_INITIAL - M) / MONOMER_INITIAL,
            0.0,
            0.999,
        )

        molecular_weight = (
            MONOMER_MOLAR_MASS
            * (Y2 + Q2)
            / (Y1 + Q1 + EPS)
        )

        kt_eff = termination_rate(
            conversion,
            params.temperature,
            molecular_weight,
            Y0,
        )

        return [
            -params.kd * I,
            -params.kp * M * Y0,
            2.0 * params.k1p * Y0 * Q1,

            2.0 * params.f * params.kd * I
            - kt_eff * Y0 * Y0,

            params.kp * M * Y0
            - kt_eff * Y0 * Y1
            + params.k1p * Y0 * Q2,

            params.kp * M * (Y0 + 2.0 * Y1)
            - kt_eff * Y0 * Y2
            + params.k1p * (2.0 * Y1 * Q2 + Y0 * q3),

            0.5 * kt_eff * Y0 * Y0
            - params.k1p * Y0 * Q1,

            kt_eff * Y0 * Y1
            - params.k1p * Y0 * Q2,

            kt_eff * Y0 * Y2
            + kt_eff * Y1 * Y1
            - params.k1p * Y0 * q3,
        ]

    initial_state = [
        params.initiator_0,
        MONOMER_INITIAL,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1e-5,
        0.0,
    ]

    solution = solve_ivp(
        rhs,
        (time_grid[0], time_grid[-1]),
        initial_state,
        method="BDF",
        t_eval=time_grid,
        rtol=1e-6,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    return solution.t, solution.y.T


# ---------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------

def calculate_observables(solution: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate conversion, molecular weights and crosslink metric."""
    monomer = solution[:, 1]
    p_q1 = solution[:, 2]

    y0 = solution[:, 3]
    y1 = solution[:, 4]
    y2 = solution[:, 5]

    q0 = solution[:, 6]
    q1 = solution[:, 7]
    q2 = solution[:, 8]

    conversion = np.clip(
        (MONOMER_INITIAL - monomer) / MONOMER_INITIAL,
        0.0,
        0.999,
    )

    mn = MONOMER_MOLAR_MASS * (y1 + q1) / (y0 + q0 + EPS)
    mw = MONOMER_MOLAR_MASS * (y2 + q2) / (y1 + q1 + EPS)

    # This definition is preserved from the original analysis block.
    crosslinks = p_q1 / (q1 + EPS)

    return {
        "conversion": conversion,
        "Mn": mn,
        "Mw": mw,
        "crosslinks": crosslinks,
    }


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_parameter_sweep(
    parameter_name: str,
    parameter_values: list[float],
    base_params: ModelParameters,
    time_grid: np.ndarray,
) -> None:
    """Run a parameter sweep and visualize the model response."""

    results = []

    for value in parameter_values:
        params_dict = base_params.__dict__.copy()
        params_dict[parameter_name] = value
        params = ModelParameters(**params_dict)

        time, solution = solve_polymerization(params, time_grid)
        observables = calculate_observables(solution)

        results.append((value, time, observables))

    # Conversion vs time
    plt.figure(figsize=(7, 5))
    for value, time, obs in results:
        plt.plot(
            time,
            obs["conversion"],
            label=f"{parameter_name} = {value}",
        )

    plt.xlabel("Time, s")
    plt.ylabel("Conversion, X")
    plt.title("Conversion vs Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Molecular weights vs conversion
    plt.figure(figsize=(7, 5))
    for value, _, obs in results:
        plt.plot(
            obs["conversion"],
            obs["Mn"],
            "--",
            label=f"Mn, {parameter_name}={value}",
        )
        plt.plot(
            obs["conversion"],
            obs["Mw"],
            label=f"Mw, {parameter_name}={value}",
        )

    plt.yscale("log")
    plt.xlabel("Conversion, X")
    plt.ylabel("Molecular Weight")
    plt.title("Molecular Weight vs Conversion")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Crosslink metric vs conversion
    plt.figure(figsize=(7, 5))
    for value, _, obs in results:
        mask = obs["conversion"] >= 0.1
        plt.plot(
            obs["conversion"][mask],
            obs["crosslinks"][mask],
            label=f"{parameter_name} = {value}",
        )

    plt.xlabel("Conversion, X")
    plt.ylabel("Crosslink Metric")
    plt.title("Crosslink Formation vs Conversion")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Example studies
# ---------------------------------------------------------------------

def main() -> None:
    """Run two representative parameter studies."""

    base_params = ModelParameters(
        f=0.5,
        kd=0.03844,
        kp=400.0,
        k1p=5.0,
        initiator_0=0.001,
        temperature=298.0,
    )

    time_grid = np.linspace(0.0, 150.0, 2000)

    # Study 1: influence of k1p
    plot_parameter_sweep(
        parameter_name="k1p",
        parameter_values=[5.0, 35.0, 45.0],
        base_params=base_params,
        time_grid=time_grid,
    )

    # Study 2: influence of initiator concentration
    initiator_params = ModelParameters(
        f=0.5,
        kd=0.03844,
        kp=500.0,
        k1p=2.0,
        initiator_0=0.01,
        temperature=298.0,
    )

    plot_parameter_sweep(
        parameter_name="initiator_0",
        parameter_values=[0.001, 0.005, 0.0085, 0.015],
        base_params=initiator_params,
        time_grid=time_grid,
    )


if __name__ == "__main__":
    main()

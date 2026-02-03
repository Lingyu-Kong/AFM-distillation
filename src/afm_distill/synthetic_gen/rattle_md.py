from __future__ import annotations

import copy
from typing_extensions import Literal

import ase
from ase.md.langevin import Langevin
from ase.md.npt import NPT
import ase.units as units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
import numpy as np
from tqdm import tqdm
import nshconfig as C

from ase.calculators.calculator import Calculator
from mattertune.wrappers.ase_calculator import MatterTuneCalculator
from .rattle_relax import min_separation_check, too_similar

GPa_to_eV_A3 = units.GPa


def perturb_cell_strain(
    atoms: ase.Atoms,
    vol_sigma: float,
    shear_sigma: float,
    max_abs_strain: float,
    max_tries: int = 1000,
):
    """
    perturb cell: H'=(I+ε)H
    - vol_sigma: volumn change
    - shear_sigma: shear change
    - max_abs_strain: upper limit for main strain
    """
    H = atoms.cell.array.copy()
    det0 = np.linalg.det(H)
    if det0 <= 0:
        raise ValueError("Input cell determinant must be positive.")
    I = np.eye(3)

    for _ in range(max_tries):
        dlogV = np.random.normal(0.0, vol_sigma)
        s = np.exp(dlogV / 3.0)
        eps_vol = (s - 1.0) * I

        A = np.random.normal(0.0, shear_sigma, size=(3, 3))
        eps = 0.5 * (A + A.T)
        eps -= np.trace(eps) / 3.0 * I

        eps_total = eps_vol + eps

        eps_total = np.clip(eps_total, -max_abs_strain, max_abs_strain)

        H_new = H @ (I + eps_total)

        det_new = np.linalg.det(H_new)
        if det_new <= 0:
            continue

        vol_ratio = det_new / det0
        if not (0.5 < vol_ratio < 2.0):
            continue

        atoms.set_cell(H_new, scale_atoms=True)
        return atoms

    raise RuntimeError(
        "Failed to generate a valid perturbed cell within max_tries. This should be abnormal, please check whether the input lattice is wrong")


def max_force_check(
    forces: np.ndarray,
    threshold: float,
):
    max_force = np.max(np.linalg.norm(forces, axis=1))
    return max_force < threshold


class SyntheticNVTConfig(C.Config):
    name: Literal["nvt"] = "nvt"

    timestep: float
    """in fs"""

    temperature_K: float

    friction: float

    fixcm: bool = True

    md_steps: int


class SyntheticNPTConfig(C.Config):
    name: Literal["npt"] = "npt"

    timestep: float
    """in fs"""

    temperature_K: float

    externalstress: float | np.ndarray | None
    """
    The external stress in eV/A^3.  Either a symmetric
    3x3 tensor, a 6-vector representing the same, or a
    scalar representing the pressure.  Note that the
    stress is positive in tension whereas the pressure is
    positive in compression: giving a scalar p is
    equivalent to giving the tensor (-p, -p, -p, 0, 0, 0).
    """

    ttime: float | None = None
    """
    Characteristic timescale of the thermostat, in ASE internal units
    Set to None to disable the thermostat.

    WARNING: Not specifying ttime sets it to None, disabling the
    thermostat.
    """

    pfactor: float | None = None
    """
    A constant in the barostat differential equation.  If
    a characteristic barostat timescale of ptime is
    desired, set pfactor to ptime^2 * B
    (where ptime is in units matching
    eV, Å, u; and B is the Bulk Modulus, given in eV/Å^3).
    Set to None to disable the barostat.
    Typical metallic bulk moduli are of the order of
    100 GPa or 0.6 eV/A^3.

    WARNING: Not specifying pfactor sets it to None, disabling the
    barostat.
    """

    mask: tuple[int] | np.ndarray | None
    """
    None or 3-tuple or 3x3 nparray (optional)
    Optional argument.  A tuple of three integers (0 or 1),
    indicating if the system can change size along the
    three Cartesian axes.  Set to (1,1,1) or None to allow
    a fully flexible computational box.  Set to (1,1,0)
    to disallow elongations along the z-axis etc.
    mask may also be specified as a symmetric 3x3 array
    indicating which strain values may change.
    """

    md_steps: int


class RattleMDSyntheticGenerator():
    """
    This synthetic data generator follows the rattle-MD pipeline for
    generating synthetic data
    """

    def __init__(
        self,
        sample_interval: int,
        min_sigma: float,
        max_sigma: float,
        vol_sigma: float = 0.02,
        shear_sigma: float = 0.02,
        max_abs_strain: float = 0.06,
        min_separation: float = 0.5,
        max_force: float = 30,
        similarity_threshold: float = 0.1,
        early_stop_patience: int = 100,
    ):
        self.sample_interval = sample_interval
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.vol_sigma = vol_sigma
        self.shear_sigma = shear_sigma
        self.max_abs_strain = max_abs_strain
        self.min_separation = min_separation
        self.max_force = max_force
        self.similarity_threshold = similarity_threshold
        self.early_stop_patience = early_stop_patience

    def _rattle_structure(
        self,
        atoms: ase.Atoms,
    ):
        if any(atoms.pbc):
            atoms = perturb_cell_strain(
                atoms=atoms,
                vol_sigma=self.vol_sigma,
                shear_sigma=self.shear_sigma,
                max_abs_strain=self.max_abs_strain,
            )

        log_lo, log_hi = np.log(self.min_sigma), np.log(self.max_sigma)
        sigma = np.exp(np.random.uniform(log_lo, log_hi))
        pos_change = np.random.randn(len(atoms), 3) * sigma
        pos_change -= np.mean(pos_change, axis=0)
        atoms.set_positions(np.array(atoms.positions) + pos_change)
        return atoms

    def generate(
        self,
        calc: MatterTuneCalculator | Calculator,
        starting_atoms: ase.Atoms,
        num_trajs: int,
        md_config: SyntheticNVTConfig | SyntheticNPTConfig,
    ):
        # rattle atoms
        pool = [copy.deepcopy(starting_atoms)]
        while len(pool) < num_trajs:
            rattled_atoms = self._rattle_structure(
                copy.deepcopy(starting_atoms),
            )
            if (
                min_separation_check(rattled_atoms, self.min_separation) and
                not too_similar(np.array(rattled_atoms.positions),
                                pool, self.similarity_threshold)
            ):
                pool.append(rattled_atoms)

        # run MD
        sampled_data: list[ase.Atoms] = []
        for idx in range(len(pool)):
            atoms = pool[idx]
            atoms.calc = calc
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=md_config.temperature_K)
            Stationary(atoms)      # remove COM drift
            ZeroRotation(atoms)    # optional
            md_steps = md_config.md_steps
            match md_config.name:
                case "nvt":
                    dyn = Langevin(
                        atoms,
                        temperature_K=md_config.temperature_K,
                        timestep=md_config.timestep * units.fs,
                        friction=md_config.friction,
                        fixcm=md_config.fixcm
                    )
                case "npt":
                    dyn = NPT(
                        atoms,
                        temperature_K=md_config.temperature_K,
                        timestep=md_config.timestep * units.fs,
                        externalstress=md_config.externalstress,  # type: ignore
                        ttime=md_config.ttime * units.fs if md_config.ttime is not None else None,
                        pfactor=md_config.pfactor,
                        mask=md_config.mask,
                    )
                case _:
                    raise ValueError(
                        f"Unknown MD thermostat: {md_config.name}")

            sampled_atoms = copy.deepcopy(atoms)
            sampled_atoms.info["traj_idx"] = idx
            e, f = sampled_atoms.get_potential_energy(), sampled_atoms.get_forces()
            sampled_data.append(sampled_atoms)
            V0 = atoms.get_volume()
            early_stop_patience = 0
            for step_i in tqdm(range(md_steps), desc=f"Running {idx+1}/{num_trajs} {md_config.name.upper()} MD, sample every {self.sample_interval} steps"):
                if step_i == 0:
                    dyn.run(1)
                else:
                    dyn.step()

                if (step_i+1) % self.sample_interval == 0:
                    # check max_force, min_separation, volume, and temperature
                    force_check = max_force_check(
                        np.array(dyn.atoms.get_forces()), self.max_force)
                    separation_check = min_separation_check(
                        dyn.atoms, self.min_separation)
                    temp_check = dyn.atoms.get_temperature() < md_config.temperature_K * 1e3
                    volumn_check = not any(dyn.atoms.pbc) or (
                        0.5 * V0 < dyn.atoms.get_volume() < 5 * V0)
                    if not force_check or not separation_check or not temp_check or not volumn_check:
                        early_stop_patience += 1
                    else:
                        early_stop_patience = 0
                    if early_stop_patience >= self.early_stop_patience:
                        break
                    if force_check and separation_check and volumn_check:
                        sampled_atoms = copy.deepcopy(dyn.atoms)
                        sampled_atoms.info["traj_idx"] = idx
                        e, f = sampled_atoms.get_potential_energy(), sampled_atoms.get_forces()
                        sampled_data.append(sampled_atoms)

        return sampled_data

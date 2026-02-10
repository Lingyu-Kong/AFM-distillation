from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Literal

import ase
import numpy as np
import vesin
from tqdm import tqdm
from ase.calculators.calculator import Calculator

from mattertune.wrappers.ase_calculator import MatterTuneCalculator


def unique_id() -> str:
    return str(uuid.uuid4())


def min_separation_check(atoms: ase.Atoms, min_separation: float) -> bool:
    (i,) = vesin.ase_neighbor_list("i", atoms, cutoff=min_separation)
    return len(i) == 0


def too_similar(pos: np.ndarray, pool: list[ase.Atoms], threshold: float) -> bool:
    for p in pool:
        if np.linalg.norm(pos - np.array(p.positions), axis=1).mean() < threshold:
            return True
    return False


def direction(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1)
    norm[norm < 1e-6] = 1e-6
    return v / norm[:, None]


def max_force_from_arrays(atoms: ase.Atoms) -> float:
    return np.linalg.norm(atoms.arrays["forces"], axis=1).max()


class RattleRelaxSyntheticGenerator:
    """
    Minimal-fix reproduction of augment-atoms rattle+relax.

    Key equivalence fixes:
    - beta mixing direction matches upstream
    - use kT (with units) rather than raw T
    - use a local RNG (RandomState) for reproducibility and equivalence
    - explicit label() writing info["energy"] + arrays["forces"] (avoid ASE cache ambiguity)
    - generate() can interpret target_num_structures as "new structures to generate"
    """

    def __init__(
        self,
        T: float,
        beta: float,
        min_sigma: float,
        max_sigma: float,
        cell_sigma: float | None = None,
        max_force: float = 30.0,
        min_separation: float = 0.5,
        max_relax_steps: int = 20,
        similarity_threshold: float = 0.1,
        # --- minimal additions to match upstream behavior ---
        seed: int = 42,
        units: Literal["eV", "kcal/mol"] = "eV",
    ):
        self.T = float(T)
        assert 0 <= beta <= 1, f"beta must be between 0 and 1, found beta={beta}"
        self.beta = float(beta)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.cell_sigma = None if cell_sigma is None else float(cell_sigma)
        self.max_force = float(max_force)
        self.min_separation = float(min_separation)
        self.max_relax_steps = int(max_relax_steps)
        self.similarity_threshold = float(similarity_threshold)

        self.seed = int(seed)
        self.units = units
        self.rand = np.random.RandomState(self.seed)

    def _get_kT(self) -> float:
        # matches augment-atoms Config.get_kT()
        if self.units == "eV":
            k = 8.617333262145e-5
        elif self.units == "kcal/mol":
            k = 0.0019872041
        else:
            raise ValueError(f"Unknown units: {self.units}")
        return self.T * k

    def _label(self, atoms: ase.Atoms, calc: MatterTuneCalculator | Calculator) -> ase.Atoms:
        """
        Explicitly compute and store energy/forces like augment-atoms label().
        Avoid relying on ASE implicit caching semantics.
        """
        atoms = atoms.copy()
        calc.calculate(atoms, ["energy", "forces"])
        atoms.arrays["forces"] = calc.results["forces"]
        atoms.info["energy"] = calc.results["energy"]
        return atoms

    def _select_parent(self, pool: list[ase.Atoms], kT: float) -> ase.Atoms:
        # use stored info["energy"] (like upstream)
        energies = np.array([atoms.info["energy"] / len(atoms) for atoms in pool])
        energy_probs = np.exp(-(energies - energies[0]) / kT)
        energy_probs /= energy_probs.sum()

        generations = np.array([atoms.info["g_i"] for atoms in pool]) + 1
        generation_probs = generations / np.sum(generations)

        # IMPORTANT: match upstream direction
        probs = (1 - self.beta) * energy_probs + self.beta * generation_probs
        probs[np.isnan(probs)] = 0
        idx = self.rand.choice(len(pool), p=probs)
        return pool[idx]

    def _rattle_structure(
        self,
        starting_atoms: ase.Atoms,
        parent: ase.Atoms,
        atoms: ase.Atoms,
    ) -> tuple[ase.Atoms, np.ndarray]:
        if self.cell_sigma is not None:
            cell_change = (
                self.rand.randn(3, 3)
                * self.cell_sigma
                * np.linalg.norm(starting_atoms.cell.array, axis=1)
            ).T
            new_cell = starting_atoms.cell.array + cell_change.T
            atoms.set_cell(new_cell, scale_atoms=True)
            restoring_matrix = starting_atoms.cell.array @ np.linalg.inv(atoms.cell.array)
        else:
            restoring_matrix = np.eye(3)

        log_lo, log_hi = np.log(self.min_sigma), np.log(self.max_sigma)
        sigma = np.exp(self.rand.uniform(log_lo, log_hi))

        pos_change = self.rand.randn(len(atoms), 3) * sigma
        pos_change -= np.mean(pos_change, axis=0)
        atoms.set_positions(np.array(atoms.positions) + pos_change)

        atoms.info.update(
            {
                "sigma": float(sigma),
                "parent": parent.info["id"],
                "g_i": parent.info["g_i"] + 1,
                "id": unique_id(),
            }
        )
        return atoms, restoring_matrix

    def _robbins_monro_relax(
        self,
        atoms: ase.Atoms,
        restoring_matrix: np.ndarray,
        calc: MatterTuneCalculator | Calculator,
        parent: ase.Atoms,
        pool: list[ase.Atoms],
        kT: float,
    ) -> ase.Atoms | None:
        atoms.info["relax_steps"] = 0

        # label once at start (upstream does label(child, calc) before loop)
        s = self._label(atoms, calc)
        prev_s = s.copy()

        for i in range(1, self.max_relax_steps + 2):
            # similarity check uses labeled structures' positions
            if too_similar(
                np.array(s.positions) @ restoring_matrix,
                pool,
                self.similarity_threshold,
            ):
                if not too_similar(
                    np.array(prev_s.positions) @ restoring_matrix,
                    pool,
                    self.similarity_threshold,
                ):
                    if max_force_from_arrays(prev_s) >= self.max_force:
                        return None
                    elif not min_separation_check(prev_s, self.min_separation):
                        return None
                    else:
                        return prev_s
                else:
                    return None

            if i == self.max_relax_steps + 1:
                if max_force_from_arrays(s) >= self.max_force:
                    return None
                elif not min_separation_check(s, self.min_separation):
                    return None
                else:
                    return s

            delta_E = (s.info["energy"] - parent.info["energy"]) / len(s)
            prob = np.exp(-delta_E / kT)
            prob = min(0.25, prob)

            if (
                self.rand.uniform() < prob
                and max_force_from_arrays(s) < self.max_force
                and min_separation_check(s, self.min_separation)
            ):
                return s

            prev_s = s.copy()
            s.info["relax_steps"] += 1

            # Robbins–Monro step: R <- R + (sigma/i) * F/||F||
            dir_vec = direction(s.arrays["forces"])
            factor = float(s.info["sigma"]) / i
            s.positions = np.array(s.positions) + factor * dir_vec

            # (Upstream does not wrap each step; keep optional. If you want strict upstream, comment next line.)
            s.wrap()

            s = self._label(s, calc)

        return None

    def generate(
        self,
        calc: MatterTuneCalculator | Calculator,
        starting_atoms: ase.Atoms,
        target_num_structures: int,
        existing_pool: list[ase.Atoms] | None = None,
    ) -> list[ase.Atoms]:
        
        kT = self._get_kT()

        existing_pool = [] if existing_pool is None else list(existing_pool)
        final_pool = list(existing_pool)

        # Ensure existing pool is labeled & has required metadata
        labeled_pool: list[ase.Atoms] = []
        for a in final_pool:
            if "energy" not in a.info or "forces" not in a.arrays:
                a = self._label(a, calc)
            if "id" not in a.info:
                a.info["id"] = unique_id()
            if "g_i" not in a.info:
                a.info["g_i"] = 0
            if "parent" not in a.info:
                a.info["parent"] = None
            if "relax_steps" not in a.info:
                a.info["relax_steps"] = 0
            labeled_pool.append(a)
        final_pool = labeled_pool

        if len(final_pool) == 0:
            s0 = starting_atoms.copy()
            if s0.pbc.any():
                s0.wrap()
            s0.info.update({"id": unique_id(), "g_i": 0, "parent": None, "relax_steps": 0})
            s0 = self._label(s0, calc)
            final_pool.append(s0)

        # termination condition
        target_total = len(final_pool) + target_num_structures

        pbar = tqdm(total=target_total, initial=len(final_pool), desc="Rattle+Relax")
        while len(final_pool) < target_total:
            parent = self._select_parent(final_pool, kT)

            child = copy.deepcopy(parent)
            child, restoring_matrix = self._rattle_structure(
                starting_atoms=starting_atoms,
                parent=parent,
                atoms=child,
            )

            relaxed = self._robbins_monro_relax(
                atoms=child,
                restoring_matrix=restoring_matrix,
                calc=calc,
                parent=parent,
                pool=final_pool,
                kT=kT,
            )
            if relaxed is not None:
                if relaxed.pbc.any():
                    relaxed.wrap()
                final_pool.append(relaxed)
                pbar.update(1)

        pbar.close()

        return final_pool[len(existing_pool):]

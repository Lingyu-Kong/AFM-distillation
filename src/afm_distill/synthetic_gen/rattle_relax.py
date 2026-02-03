from __future__ import annotations

import copy
import uuid

import ase
import numpy as np
import vesin
from tqdm import tqdm

from ase.calculators.calculator import Calculator
from mattertune.wrappers.ase_calculator import MatterTuneCalculator


def unique_id():
    return str(uuid.uuid4())


def min_separation_check(
    atoms: ase.Atoms,
    min_separation: float,
):
    (i,) = vesin.ase_neighbor_list(
        "i", atoms, cutoff=min_separation
    )
    return len(i) == 0


def too_similar(
    pos: np.ndarray,
    pool: list[ase.Atoms],
    threshold: float
):
    for p in pool:
        if np.linalg.norm(pos - np.array(p.positions), axis=1).mean() < threshold:
            return True
    return False


class RattleRelaxSyntheticGenerator():
    """
    This synthetic data generator follows the rattle-relax pipeline for 
    generating synthetic data introduced in https://arxiv.org/abs/2506.10956 and implemented in https://github.com/jla-gardner/augment-atoms
    """

    def __init__(
        self,
        T: float,
        beta: float,
        min_sigma: float,
        max_sigma: float,
        cell_sigma: float | None = None,
        max_force: float = 30,
        min_separation: float = 0.5,
        max_relax_steps: int = 20,
        similarity_threshold: float = 0.1,
    ):
        self.T = T
        assert 0 <= beta <= 1, f"beta must be between 0 and 1, found beta={beta}"
        self.beta = beta
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.cell_sigma = cell_sigma
        self.max_force = max_force
        self.min_separation = min_separation
        self.max_relax_steps = max_relax_steps
        self.similarity_threshold = similarity_threshold

    def _select_parent(
        self,
        pool: list[ase.Atoms],
    ):
        energies = np.array(
            [atoms.get_potential_energy()/len(atoms) for atoms in pool])
        energy_probs = np.exp(-(energies-energies[0])/self.T)
        energy_probs /= np.sum(energy_probs)

        generations = np.array([atoms.info["g_i"] for atoms in pool]) + 1
        generation_probs = generations / np.sum(generations)

        probs = self.beta * energy_probs + (1 - self.beta) * generation_probs
        probs[np.isnan(probs)] = 0
        idx = np.random.choice(len(pool), p=probs)
        return pool[idx]

    def _rattle_structure(
        self,
        starting_atoms: ase.Atoms,
        parent: ase.Atoms,
        atoms: ase.Atoms,
    ):
        if self.cell_sigma is not None:
            cell_change = (
                np.random.randn(3, 3)
                * self.cell_sigma
                * np.linalg.norm(starting_atoms.cell.array, axis=1)
            ).T
            new_cell = starting_atoms.cell.array + cell_change.T
            atoms.set_cell(new_cell, scale_atoms=True)
            restoring_matrix = starting_atoms.cell.array @ np.linalg.inv(
                atoms.cell.array
            )
        else:
            restoring_matrix = np.eye(3)

        log_lo, log_hi = np.log(self.min_sigma), np.log(self.max_sigma)
        sigma = np.exp(np.random.uniform(log_lo, log_hi))
        pos_change = np.random.randn(len(atoms), 3) * sigma
        pos_change -= np.mean(pos_change, axis=0)
        atoms.set_positions(np.array(atoms.positions) + pos_change)
        atoms.info.update({
            "sigma": sigma,
            "parent": parent.info["id"],
            "g_i": parent.info["g_i"] + 1,
            "id": unique_id(),
        })
        return atoms, restoring_matrix

    def _robbins_monro_relax(
        self,
        atoms: ase.Atoms,
        restoring_matrix: np.ndarray,
        calc: MatterTuneCalculator | Calculator,
        parent: ase.Atoms,
        pool: list[ase.Atoms]
    ):
        atoms.info["relax_steps"] = 0
        atoms.calc = calc
        prev_s = copy.deepcopy(atoms)

        for i in range(1, self.max_relax_steps + 2):
            if too_similar(
                np.array(atoms.positions) @ restoring_matrix,
                pool,
                self.similarity_threshold,
            ):
                if not too_similar(
                    np.array(prev_s.positions) @ restoring_matrix,
                    pool,
                    self.similarity_threshold,
                ):
                    if np.linalg.norm(prev_s.get_forces(), axis=1).max() >= self.max_force:
                        return None
                    elif not min_separation_check(prev_s, self.min_separation):
                        return None
                    else:
                        return prev_s
                else:
                    return None

                break

            if i == self.max_relax_steps + 1:
                if np.linalg.norm(atoms.get_forces(), axis=1).max() >= self.max_force:
                    return None
                elif not min_separation_check(atoms, self.min_separation):
                    return None
                else:
                    return atoms

                break

            delta_E = (atoms.get_potential_energy() -
                       parent.get_potential_energy()) / len(atoms)
            prob = np.exp(-delta_E / self.T)
            prob = min(0.25, prob)
            if (
                np.random.uniform() < prob and
                np.linalg.norm(atoms.get_forces(), axis=1).max() < self.max_force and
                min_separation_check(atoms, self.min_separation)
            ):
                return atoms

            prev_s = copy.deepcopy(atoms)
            atoms.info["relax_steps"] += 1
            forces = atoms.get_forces()
            norm = np.linalg.norm(forces, axis=1)
            norm[norm < 1e-6] = 1e-6
            forces = forces / norm[:, None]
            factor = atoms.info["sigma"] / i
            atoms.set_positions(np.array(atoms.positions) + factor * forces)
            atoms.wrap()
            atoms.calc = calc

    def generate(
        self,
        calc: MatterTuneCalculator | Calculator,
        starting_atoms: ase.Atoms,
        target_num_structures: int,
        existing_pool: list[ase.Atoms] | None = None,
    ):
        """
        Each Atoms object has an additional generation idx:
        atoms.info["g_i"]
        """
        pbar = tqdm(range(target_num_structures),
                    desc="Generating Synthetic Data by Rattling+Relaxation")
        final_pool = existing_pool if existing_pool is not None else []
        for atoms in final_pool:
            atoms.calc = calc
        if len(final_pool) == 0:
            # put starting atoms into pool
            starting_atoms.info.update({
                "id": unique_id(),
                "g_i": 0,
                "parent": None,
                "relax_steps": 0,
            })
            starting_atoms.wrap()
            starting_atoms.calc = calc
            final_pool.append(starting_atoms)
        for _ in final_pool:
            pbar.update(1)

        while len(final_pool) < target_num_structures:
            parent = self._select_parent(final_pool)

            child = copy.deepcopy(parent)

            child, restoring_matrix = self._rattle_structure(
                starting_atoms=starting_atoms,
                parent=parent,
                atoms=child,
            )

            relaxed_atoms = self._robbins_monro_relax(
                atoms=child,
                restoring_matrix=restoring_matrix,
                calc=calc,
                parent=parent,
                pool=final_pool,
            )
            if relaxed_atoms is not None:
                relaxed_atoms.wrap()
                final_pool.append(relaxed_atoms)
                pbar.update(1)

        return final_pool

from __future__ import annotations

import copy
import time
import rich

import ase.units as units
from ase import Atoms
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.npt import NPT
import wandb
from tqdm import tqdm
from nequip.ase import NequIPCalculator

supercell_as = [1, 2, 3, 4, 5, 8, 10, 12, 14, 16]


def main(args_dict: dict):

    md_atoms_base = read("./data/li3po4_192.xyz")
    assert isinstance(md_atoms_base, Atoms), "Expected an Atoms object"
    md_atoms_base.pbc = True

    calculator = NequIPCalculator.from_compiled_model(
        compile_path=args_dict["model"],
        device="cuda",
        chemical_species_to_atom_type_map=True,
    )
    model_type = args_dict["model"].split("/")[-1].split(".")[0]
    if args_dict["model"].endswith(".pt2"):
        model_type += "-aotinductor"
    else:
        model_type += "-torchscript"

    wandb.init(
        project="Distill-MDSpeedTest-SingleGPU",
        name=f"{model_type}-li3po4-600K-{args_dict['thermo_state']}-{args_dict['nl_fn_type']}-Skin{args_dict['skin_cutoff']}A",
        save_code=False,
    )
    wandb.config.update(args_dict)

    for a in supercell_as:
        md_atoms = copy.deepcopy(md_atoms_base)
        md_atoms = md_atoms * (a, a, a)
        md_atoms.calc = calculator
        rich.print(f"Running supercell {a}x{a}x{a}, {len(md_atoms)} atoms")

        if args_dict["thermo_state"].lower() == "nvt":
            dyn = Langevin(
                md_atoms,
                temperature_K=600,
                timestep=args_dict["timestep"] * units.fs,
                friction=args_dict["friction"],
            )
        elif args_dict["thermo_state"].lower() == "npt":
            dyn = NPT(
                md_atoms,
                temperature_K=600,
                timestep=args_dict["timestep"] * units.fs,
                externalstress=None,
                ttime=100 * units.fs,
                pfactor=None,
            )
        else:
            raise ValueError(
                "Invalid thermo_state, must be one of 'NVT' or 'NPT'")

        time1 = time.time()
        try:
            for step_i in tqdm(range(args_dict["steps"])):
                if step_i == 0:
                    dyn.run(1)
                else:
                    dyn.step()
                wandb.log({
                    "Temperature (K)": dyn.atoms.get_temperature(),
                })
        except Exception as e:
            # check if it is CUDA out of memory error
            if "CUDA out of memory" in str(e):
                rich.print(
                    f"[red]CUDA out of memory for supercell {a}x{a}x{a}, skipping...[/red]")
                continue
            else:
                raise e
        time2 = time.time()

        ps_per_hour = (
            args_dict["steps"] * args_dict["timestep"]) / (time2 - time1) * 3600 / 1000
        rich.print(
            f"Supercell {a}x{a}x{a}, {len(md_atoms)} atoms, {ps_per_hour:.2f} ps/hour")
        wandb.log({f"Speed (ps/hour) {a}x{a}x{a}": ps_per_hour})
        rich.print("=" * 50)

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="./checkpoints/allegro-4.0A-T=1.nequip.pt2")
    parser.add_argument("--thermo_state", type=str, default="NVT")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timestep", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--nl_fn_type", type=str, default=None)
    parser.add_argument("--skin_cutoff", type=float, default=None)
    args_dict = vars(parser.parse_args())
    main(args_dict)

from __future__ import annotations

import copy
import os
import argparse
import rich
import datetime

from ase import Atoms
from ase.io import read, write

import mattertune.configs as MC
from mattertune.main import load_pretrained_model, load_finetuned_checkpoint
from mattertune.util import optional_import_error_message
from afm_distill.synthetic_gen.rattle_md import RattleMDSyntheticGenerator, SyntheticNVTConfig

import wandb


def main(args_dict: dict):
    model_path = args_dict["model_path"]
    if model_path.endswith(".ckpt"):
        model_name = model_path.split("/")[-1].split(".")[0]
        model = load_finetuned_checkpoint(model_path)
        calc = model.ase_calculator(device=f"cuda:{args_dict['device']}")
    else:
        model_name = model_path
        if "mattersim" in model_path.lower():
            model_config = MC.MatterSimBackboneConfig.draft()
            model_config.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
            model_config.pretrained_model = model_path
            model = load_pretrained_model(model_config)
            calc = model.ase_calculator(device=f"cuda:{args_dict['device']}")
        elif "mace" in model_path.lower():
            model_config = MC.MACEBackboneConfig.draft()
            model_config.pretrained_model = model_path
            model = load_pretrained_model(model_config)
            calc = model.ase_calculator(device=f"cuda:{args_dict['device']}")
        elif "nequip" in model_path.lower():
            model_config = MC.NequIPBackboneConfig.draft()
            model_config.pretrained_model = model_path
            model = load_pretrained_model(model_config)
            calc = model.ase_calculator(device=f"cuda:{args_dict['device']}")
        elif "uma" in model_path.lower():
            with optional_import_error_message("fairchem"):
                from fairchem.core import FAIRChemCalculator
                from fairchem.core import pretrained_mlip
            predictor = pretrained_mlip.get_predict_unit(
                model_path, device="cuda", inference_settings="default")
            calc = FAIRChemCalculator(
                predictor,
                task_name="omat",  # options: "omol", "omat", "odac", "oc20", "omc"
            )
        else:
            raise ValueError(f"Unknown model type: {model_path}")

    starting_atoms_list: list[Atoms] = read(
        args_dict["starting_structrues"], index=":")  # type:ignore

    wandb.init(
        project="Distill-SyntheticGen",
        name=f"RattleMD-{model_name}-{args_dict['starting_structrues'].split('/')[-1].split('.')[0]}",
        save_code=False,
    )
    wandb.config.update(args_dict)

    sync_generator = RattleMDSyntheticGenerator(
        sample_interval=args_dict["sample_interval"],
        min_sigma=args_dict["min_sigma"],
        max_sigma=args_dict["max_sigma"],
        vol_sigma=args_dict["vol_sigma"],
        shear_sigma=args_dict["shear_sigma"],
        max_abs_strain=args_dict["max_abs_strain"],
        min_separation=args_dict["min_separation"],
        max_force=args_dict["max_force"],
        similarity_threshold=args_dict["similarity_threshold"],
        early_stop_patience=args_dict["early_stop_patience"],
    )

    save_dir = args_dict["save_dir"]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"RattleMD-{timestamp}-{model_name}-{args_dict['starting_structrues'].split('/')[-1].split('.')[0]}"
    save_path = os.path.join(save_dir, run_name)
    os.system(f"rm -rf {save_path}")
    os.makedirs(save_path)

    md_config = SyntheticNVTConfig(
        timestep=args_dict["timestep"],
        temperature_K=args_dict["temperature_K"],
        friction=args_dict["friction"],
        md_steps=args_dict["md_steps"],
    )

    total_nums_data = 0
    for idx, atoms in enumerate(starting_atoms_list):
        sampled_data = sync_generator.generate(
            calc=calc,
            starting_atoms=copy.deepcopy(atoms),
            num_trajs=args_dict["num_trajs"],
            md_config=md_config,
        )
        write(os.path.join(save_path, f"rattle_md_{idx}.xyz"), sampled_data)
        total_nums_data += len(sampled_data)
        wandb.log({"total_nums_data": total_nums_data})
        rich.print(f"Finished trajectory {idx+1}/{len(starting_atoms_list)}, total nums data: {total_nums_data}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="MatterSim-v1.0.0-1M")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--starting_structrues",
        type=str,
        default="./data/water_30.xyz",
    )
    parser.add_argument("--save_dir", type=str,
                        default="/net/csefiles/coc-fung-cluster/lingyu/afm_distill/")
    parser.add_argument("--num_trajs", type=int, default=4)
    parser.add_argument("--sample_interval", type=int, default=50)
    parser.add_argument("--min_sigma", type=float, default=0.1)
    parser.add_argument("--max_sigma", type=float, default=0.5)
    parser.add_argument("--vol_sigma", type=float, default=0.02)
    parser.add_argument("--shear_sigma", type=float, default=0.02)
    parser.add_argument("--max_abs_strain", type=float, default=0.06)
    parser.add_argument("--min_separation", type=float, default=0.5)
    parser.add_argument("--max_force", type=float, default=30.0)
    parser.add_argument("--similarity_threshold", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=100)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--temperature_K", type=float, default=600.0)
    parser.add_argument("--friction", type=float, default=0.02)
    parser.add_argument("--md_steps", type=int, default=50000)
    return parser


if __name__ == "__main__":
    import argparse

    parser = build_arg_parser()
    args = parser.parse_args()
    main(vars(args))

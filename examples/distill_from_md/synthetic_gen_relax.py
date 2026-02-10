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
from afm_distill.synthetic_gen.rattle_relax import RattleRelaxSyntheticGenerator

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
        name=f"RattleRelax-{model_name}-{args_dict['starting_structrues'].split('/')[-1].split('.')[0]}",
        save_code=False,
    )
    wandb.config.update(args_dict)
    
    sync_generator = RattleRelaxSyntheticGenerator(
        T=args_dict["T"],
        beta=args_dict["beta"],
        min_sigma=args_dict["min_sigma"],
        max_sigma=args_dict["max_sigma"],
        cell_sigma=args_dict["cell_sigma"],
        max_force=args_dict["max_force"],
        min_separation=args_dict["min_separation"],
        max_relax_steps=args_dict["max_relax_steps"],
        similarity_threshold=args_dict["similarity_threshold"],
    )
    
    save_dir = args_dict["save_dir"]
    # date time year-month-day-hour-minute
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"RattleRelax-{timestamp}-{model_name}-{args_dict['starting_structrues'].split('/')[-1].split('.')[0]}"
    save_path = os.path.join(save_dir, run_name)
    os.system(f"rm -rf {save_path}")
    os.makedirs(save_path)
    
    total_nums_data = 0
    for idx, atoms in enumerate(starting_atoms_list):
        sampled_data = sync_generator.generate(
            calc = calc,
            starting_atoms= atoms,
            target_num_structures=args_dict["n_per_structure"]
        )
        write(os.path.join(save_path, f"rattle_relax_{idx}.xyz"), sampled_data)
        total_nums_data += len(sampled_data)
        wandb.log({"total_nums_data": total_nums_data})
        rich.print(f"Generated {len(sampled_data)} structures for starting structure {idx}, total generated data: {total_nums_data}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="MatterSim-v1.0.0-1M")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--starting_structrues",
        type=str,
        default="./data/h2o_1593_train_25.xyz",
    )
    parser.add_argument("--save_dir", type=str,
                        default="/net/csefiles/coc-fung-cluster/lingyu/afm_distill/")
    parser.add_argument("--T", type=float, default=300.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--min_sigma", type=float, default=0.01)
    parser.add_argument("--max_sigma", type=float, default=0.1)
    parser.add_argument("--cell_sigma", type=float, default=None)
    parser.add_argument("--max_force", type=float, default=30)
    parser.add_argument("--min_separation", type=float, default=0.5)
    parser.add_argument("--max_relax_steps", type=int, default=20)
    parser.add_argument("--similarity_threshold", type=float, default=0.1)
    parser.add_argument("--n_per_structure", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
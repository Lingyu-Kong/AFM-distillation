from __future__ import annotations

import rich

import ase
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import mattertune.configs as MC
from mattertune.main import load_pretrained_model, load_finetuned_checkpoint
from mattertune.util import optional_import_error_message
from tqdm import tqdm
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
            predictor = pretrained_mlip.get_predict_unit(model_path, device="cuda", inference_settings="default")
            calc = FAIRChemCalculator(
                predictor,
                task_name=args_dict["task_name"],
            )
        else:
            raise ValueError(f"Unknown model type: {model_path}")

    dataset_path = args_dict["dataset_path"]
    # dataset_path = "~/MatterTune/examples/water-thermodynamics/data/val_water_1000_eVAng.xyz"
    # dataset_path = "/net/csefiles/coc-fung-cluster/lingyu/datasets/Li3PO4/li3po4-test_quench.xyz"
    atoms_list: list[ase.Atoms] = read(dataset_path, ":") # type:ignore
    if args_dict["limit"] is not None and len(atoms_list) > args_dict["limit"]:
        indices = np.random.choice(len(atoms_list), args_dict["limit"], replace=False)
        atoms_list = [atoms_list[i] for i in indices]
    
    wandb.init(
        project="Distill-TeacherTest",
        name=f"{model_name}-{dataset_path.split('/')[-1].split('.')[0]}",
        save_code=False,
    )
    wandb.config.update(args_dict)

    gt_energies = []
    gt_forces = []
    pred_energies = []
    pred_forces = []
    num_atoms = []
    for atoms in tqdm(atoms_list):
        num_atoms.append(len(atoms))
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        gt_energies.append(energy)
        gt_forces.append(forces)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        pred_energies.append(energy)
        pred_forces.append(forces)
    
    gt_energies = np.array(gt_energies)
    gt_forces = np.vstack(gt_forces)
    pred_energies = np.array(pred_energies)
    pred_forces = np.vstack(pred_forces)
    num_atoms = np.array(num_atoms)
    e_mae = np.mean(np.abs(gt_energies - pred_energies))
    f_mae = np.mean(np.abs(gt_forces - pred_forces))
    e_rmse = np.sqrt(np.mean((gt_energies - pred_energies)**2))
    f_rmse = np.sqrt(np.mean((gt_forces - pred_forces)**2))
    rich.print(f"Energy MAE: {e_mae} eV")
    rich.print(f"Forces MAE: {f_mae} eV/Ang")
    rich.print(f"Energy RMSE: {e_rmse} eV")
    rich.print(f"Forces RMSE: {f_rmse} eV/Ang")
    
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.scatter(gt_energies, pred_energies, alpha=0.5)
    plt.xlabel("DFT Energy (eV)")
    plt.ylabel("Predicted Energy (eV)")
    plt.title("Energy Prediction")
    plt.plot([min(gt_energies), max(gt_energies)], [
             min(gt_energies), max(gt_energies)], 'r--')
    plt.subplot(2, 2, 2)
    plt.scatter(gt_forces[:, 0], pred_forces[:, 0], alpha=0.5)
    plt.xlabel("DFT Forces X (eV/Ang)")
    plt.ylabel("Predicted Forces X (eV/Ang)")
    plt.title("Forces X Prediction")
    plt.plot([min(gt_forces[:, 0]), max(gt_forces[:, 0])], [
             min(gt_forces[:, 0]), max(gt_forces[:, 0])], 'r--')
    plt.subplot(2, 2, 3)
    plt.scatter(gt_forces[:, 1], pred_forces[:, 1], alpha=0.5)
    plt.xlabel("DFT Forces Y (eV/Ang)")
    plt.ylabel("Predicted Forces Y (eV/Ang)")
    plt.title("Forces Y Prediction")
    plt.plot([min(gt_forces[:, 1]), max(gt_forces[:, 1])], [
             min(gt_forces[:, 1]), max(gt_forces[:, 1])], 'r--')
    plt.subplot(2, 2, 4)
    plt.scatter(gt_forces[:, 2], pred_forces[:, 2], alpha=0.5)
    plt.xlabel("DFT Forces Z (eV/Ang)")
    plt.ylabel("Predicted Forces Z (eV/Ang)")
    plt.title("Forces Z Prediction")
    plt.plot([min(gt_forces[:, 2]), max(gt_forces[:, 2])], [
             min(gt_forces[:, 2]), max(gt_forces[:, 2])], 'r--')
    plt.tight_layout()
    plt.savefig(f"{model_name}_{dataset_path.split('/')[-1].split('.')[0]}.png", dpi=300)
    plt.close()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="MatterSim-v1.0.0-1M")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task_name", type=str, default=None)
    # parser.add_argument("--dataset_path", type=str, default="/nethome/lkong88/MatterTune/examples/water-thermodynamics/data/val_water_1000_eVAng.xyz")
    parser.add_argument("--dataset_path", type=str, default="/net/csefiles/coc-fung-cluster/lingyu/datasets/Li3PO4/li3po4-test_quench.xyz")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
        
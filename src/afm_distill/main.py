"""
This file is the main entry point for the AFM distillation framework.

We implement the student training part.
"""

from __future__ import annotations

import logging
from typing import Any

import nshconfig as C
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from pydantic import model_validator

from mattertune.main import TrainerConfig as LightningTrainerConfig
from mattertune.data import DataModuleConfig, MatterTuneDataModule
from .registry import student_registry, data_registry
from .models.base import StudentModuleBase, StudentModuleBaseConfig
from .models import StudentModelConfig

log = logging.getLogger(__name__)


@student_registry.rebuild_on_registers
@data_registry.rebuild_on_registers
class OfflineDistillationConfig(C.Config):
    """
    Offline distillation framework for training student models using a pre-trained foundation model.
    A general workflow is as follows:
    1. Generate and label synthetic data using the foundation model.
    2. Train the student model using the labeled synthetic data.

    To avoid env conficts between foundation model and student model,
    the data generation part and the student training part are separated.

    Here, we implement the student training part.
    """
    data: DataModuleConfig
    """The configuration for the data."""

    model: StudentModelConfig
    """The configuration for the model."""

    trainer: LightningTrainerConfig = LightningTrainerConfig()
    """The configuration for the trainer."""

    finetune_ckpt_path: str | None = None
    """Optional path to a student model checkpoint to initialize from before fine-tuning."""

    @model_validator(mode="before")
    @classmethod
    def _merge_ckpt_model_hparams(cls, values: Any):
        if not isinstance(values, dict):
            return values
        ckpt_path = values.get("finetune_ckpt_path")
        if not ckpt_path:
            return values
        model = values.get("model")
        if model is None:
            return values

        if hasattr(model, "model_dump"):
            model_dict = model.model_dump(
                round_trip=True,
                exclude_unset=True,
            )
        elif isinstance(model, dict):
            model_dict = model
        else:
            return values

        ckpt_hparams = _load_hparams_from_checkpoint(ckpt_path)
        if not ckpt_hparams:
            return values

        ckpt_name = ckpt_hparams.get("name")
        model_name = model_dict.get("name")
        if ckpt_name is not None and model_name is not None and ckpt_name != model_name:
            raise ValueError(
                "Checkpoint model name does not match the configured model. "
                f"ckpt='{ckpt_name}', config='{model_name}'."
            )

        merged_model = dict(ckpt_hparams)
        merged_model.update(model_dict)
        values["model"] = merged_model
        return values


class OfflineDistillation:
    """
    Offline distillation framework for training student models using a pre-trained foundation model.
    A general workflow is as follows:
    1. Generate and label synthetic data using the foundation model.
    2. Train the student model using the labeled synthetic data.

    To avoid env conficts between foundation model and student model,
    the data generation part and the student training part are separated.

    Here, we implement the student training part.
    """

    def __init__(
        self,
        config: OfflineDistillationConfig,
    ):
        self.config = config

    def train(
            self,
            trainer_kwargs: dict[str, Any] | None = None) -> tuple[StudentModuleBase, Trainer]:
        ckpt_path = getattr(self.config, "finetune_ckpt_path", None)

        # Create the datamodule (needed early for CACE lazy init)
        datamodule = MatterTuneDataModule(self.config.data)

        # Resolve the full trainer kwargs
        trainer_kwargs_: dict[
            str, Any] = self.config.trainer._to_lightning_kwargs()

        if trainer_kwargs is not None:
            trainer_kwargs_.update(trainer_kwargs)

        # set up the data
        try:
            datamodule.prepare_data()
        except Exception:
            pass
        datamodule.setup(stage="fit")

        # Make sure all the necessary dependencies are installed
        self.config.model.ensure_dependencies()

        if ckpt_path:
            ckpt_model_name = _get_model_name_from_checkpoint(ckpt_path)
            config_model_name = getattr(self.config.model, "name", None)
            if config_model_name is not None and config_model_name != ckpt_model_name:
                raise ValueError(
                    "Checkpoint model name does not match the configured model. "
                    f"ckpt='{ckpt_model_name}', config='{config_model_name}'."
                )

            load_kwargs: dict[str, Any] = {}
            if ckpt_model_name == "cace":
                load_kwargs["lazy_init_atoms"] = _sample_atoms_for_lazy_init(
                    datamodule)

            lightning_module = load_model_from_checkpoint(
                ckpt_path,
                weights_only=False,
                **load_kwargs,
            )

            _maybe_override_finetune_hparams(
                self.config.model,
                lightning_module,
            )
            _reset_finetune_ckpt_path(self.config, lightning_module)
        else:
            # Create the model
            lightning_module = self.config.model.create_model()

        assert isinstance(
            lightning_module, StudentModuleBase
        ), f"Model must be an instance of StudentModuleBase, but got {type(lightning_module)}"

        if lightning_module.requires_disabled_inference_mode():
            if (user_inference_mode := trainer_kwargs_.get("inference_mode")
                ) is not None and user_inference_mode:
                raise ValueError(
                    "The model requires inference_mode to be disabled. "
                    "But the provided trainer kwargs have inference_mode=True. "
                    "Please set inference_mode=False.\n"
                    "If you think this is a mistake, please report a bug.")

            log.info("The model requires inference_mode to be disabled. "
                     "Setting inference_mode=False.")
            trainer_kwargs_["inference_mode"] = False

        # Set up the callbacks for recipes
        callbacks: list[Callback] = trainer_kwargs_.pop("callbacks", [])
        trainer_kwargs_["callbacks"] = callbacks

        lightning_module.before_fit_start(datamodule)

        # Create the trainer
        trainer = Trainer(**trainer_kwargs_)
        trainer.fit(lightning_module, datamodule)

        return lightning_module, trainer


def load_model_from_checkpoint(
    ckpt_path: str,
    weights_only: bool = False,
    **kwargs: Any,  # for some models, additional args may needed
):
    from .models import (
        SchNetStudentModel,
        PaiNNStudentModel,
        CACEStudentModel,
        AllegroStudentModel
    )

    name = _get_model_name_from_checkpoint(ckpt_path)

    match name:
        case "allegro":
            return AllegroStudentModel.load_from_checkpoint(ckpt_path, weights_only=weights_only, **kwargs)
        case "schnet":
            return SchNetStudentModel.load_from_checkpoint(ckpt_path, weights_only=weights_only, **kwargs)
        case "painn":
            return PaiNNStudentModel.load_from_checkpoint(ckpt_path, weights_only=weights_only, **kwargs)
        case "cace":
            return CACEStudentModel.load_from_checkpoint(ckpt_path, weights_only=weights_only, **kwargs)
        case _:
            raise ValueError(f"Unknown model name '{name}' in checkpoint.")


def _get_model_name_from_checkpoint(ckpt_path: str) -> str:
    ckpt_hparams = _load_hparams_from_checkpoint(ckpt_path)
    name = ckpt_hparams.get("name", None)
    if name is None:
        raise ValueError(
            "Could not find model name in checkpoint hyper_parameters. "
            "Please ensure the checkpoint was saved using this afm_distill package."
        )
    return name


def _load_hparams_from_checkpoint(ckpt_path: str) -> dict[str, Any]:
    import torch

    ckpt_dict = torch.load(ckpt_path, weights_only=False)
    hparams = ckpt_dict.get("hyper_parameters", {})
    if not isinstance(hparams, dict):
        return {}
    return hparams


def _sample_atoms_for_lazy_init(datamodule: MatterTuneDataModule):
    from ase import Atoms

    dataset = datamodule.datasets.get("train")
    if dataset is None:
        raise ValueError("No training dataset found for lazy initialization.")

    def extract_atoms(item):
        if isinstance(item, Atoms):
            return item
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], Atoms):
            return item[0]
        if isinstance(item, dict) and isinstance(item.get("atoms"), Atoms):
            return item["atoms"]
        raise TypeError("Dataset items must contain an ase.Atoms object.")

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        if len(dataset) == 0:  # type: ignore[arg-type]
            raise ValueError(
                "Training dataset is empty; cannot lazy initialize.")
        return extract_atoms(dataset[0])  # type: ignore[index]

    if hasattr(dataset, "__iter__"):
        for item in dataset:
            return extract_atoms(item)
        raise ValueError("Training dataset is empty; cannot lazy initialize.")

    raise TypeError(
        "Train dataset must be indexable or iterable to provide an ase.Atoms sample."
    )


def _reset_finetune_ckpt_path(config: OfflineDistillationConfig, lightning_module: StudentModuleBase) -> None:
    if getattr(config, "finetune_ckpt_path", None) is not None:
        try:
            config.finetune_ckpt_path = None
        except Exception:
            log.warning(
                "Failed to reset finetune_ckpt_path on config; it may be immutable.")

    if hasattr(lightning_module, "hparams"):
        hparams = getattr(lightning_module, "hparams")
        if isinstance(hparams, dict) and "finetune_ckpt_path" in hparams:
            hparams["finetune_ckpt_path"] = None
        elif hasattr(hparams, "finetune_ckpt_path"):
            try:
                setattr(hparams, "finetune_ckpt_path", None)
            except Exception:
                log.warning(
                    "Failed to reset finetune_ckpt_path on model hparams.")


def _maybe_override_finetune_hparams(
    config_model: StudentModuleBaseConfig,
    lightning_module: StudentModuleBase,
) -> None:
    if not hasattr(lightning_module, "hparams"):
        return

    hparams = getattr(lightning_module, "hparams")
    fields_set = getattr(config_model, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(config_model, "__fields_set__", None)

    def set_hparam(key: str, value: Any) -> None:
        if isinstance(hparams, dict):
            hparams[key] = value
            return
        if hasattr(hparams, key):
            try:
                setattr(hparams, key, value)
            except Exception:
                log.warning("Failed to override hparam '%s' on model.", key)

    def get_hparam(key: str):
        if isinstance(hparams, dict):
            return hparams.get(key)
        return getattr(hparams, key, None)

    if (fields_set is None or "optimizer" in fields_set) and getattr(
            config_model, "optimizer", None) is not None:
        set_hparam("optimizer", config_model.optimizer)

    if (fields_set is None or "lr_scheduler" in fields_set) and hasattr(
            config_model, "lr_scheduler") and config_model.lr_scheduler is not None:
        set_hparam("lr_scheduler", config_model.lr_scheduler)

    refresh_metrics = False
    refresh_normalizers = False

    if (fields_set is None or "properties" in fields_set) and getattr(
            config_model, "properties", None):
        new_props = config_model.properties
        old_props = get_hparam("properties")
        if old_props is not None:
            old_names = _property_names(old_props)
            new_names = _property_names(new_props)
            if not set(new_names).issubset(set(old_names)):
                raise ValueError(
                    "finetune properties must be a subset of checkpoint properties. "
                    f"ckpt={sorted(set(old_names))}, new={sorted(set(new_names))}."
                )
        set_hparam("properties", new_props)
        refresh_metrics = True
        refresh_normalizers = True

    if (fields_set is None or "normalizers" in fields_set) and hasattr(
            config_model, "normalizers") and config_model.normalizers is not None and len(config_model.normalizers) > 0:
        set_hparam("normalizers", config_model.normalizers)
        refresh_normalizers = True

    if refresh_metrics or refresh_normalizers:
        try:
            if refresh_metrics:
                lightning_module.create_metrics()
            if refresh_normalizers:
                lightning_module.create_normalizers()
        except Exception:
            log.warning(
                "Failed to refresh metrics/normalizers after hparams override.")


def _property_names(props: Any) -> list[str]:
    names: list[str] = []
    for prop in props:
        name = getattr(prop, "name", None)
        if name is None and isinstance(prop, dict):
            name = prop.get("name")
        if name is None:
            raise ValueError("Property config is missing 'name'.")
        names.append(str(name))
    return names

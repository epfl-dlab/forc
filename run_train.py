from src.utils import hydra_custom_resolvers
from src import utils
import hydra
from omegaconf import OmegaConf, DictConfig

from src.utils import general_helpers
from typing import List

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import Logger

log = utils.get_pylogger(__name__)


def run_train(cfg: DictConfig):
    assert cfg.output_dir is not None, "Path to the directory in which the predictions will be written must be given"
    cfg.output_dir = general_helpers.get_absolute_path(cfg.output_dir)
    log.info(f"Output directory: {cfg.output_dir}")

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating data module <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    log.info(f"Instantiating model <{cfg.model.meta_model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model.meta_model, datamodule=datamodule)
    datamodule.set_tokenizer(model.tokenizer)
    # If defined, use the model's collate function (otherwise proceed with the PyTorch's default collate_fn)
    if getattr(model, "collator", None):
        datamodule.set_collate_fn(model.collator.collate_fn)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = general_helpers.instantiate_callbacks(cfg.get("callback"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = general_helpers.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    logging_object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(logging_object_dict)

    log.info("Starting training!")
    model.output_dir = cfg.output_dir
    if cfg.resume_from_checkpoint:
        log.info(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.resume_from_checkpoint)

    ckpt_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best ckpt path: {ckpt_path}") # TODO add option to save_pretrained to be consistent with huggingface

    if cfg.get("test"):
        log.info("Starting testing!")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        model.output_dir = cfg.output_dir
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


    metric_dict = trainer.callback_metrics
    log.info("Metrics dict:")
    log.info(metric_dict)


@hydra.main(version_base="1.2", config_path="configs", config_name="train_root")
def main(hydra_config: DictConfig):
    utils.run_task(hydra_config, run_train)


if __name__ == "__main__":
    main()
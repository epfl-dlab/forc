from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from pathlib import Path
import os
from pytorch_lightning.loggers import WandbLogger


class EvaluationUtils(object):

    @staticmethod
    def upload_outputs_to_wandb(hparams_to_log, output_dir, logger):
        # if isinstance(logger, LoggerCollection):
        #     loggers = logger
        # else:
        loggers = [logger]

        for logger in loggers:
            if isinstance(logger, WandbLogger):
                output_files = os.listdir(output_dir)
                output_files = [os.path.relpath(os.path.join(output_dir, f)) for f in output_files]

                logger.experiment.save(f"{output_dir}/*", base_path=".", policy="now")
                logger.experiment.config["output_files"] = output_files
                logger.experiment.config.update(hparams_to_log, allow_val_change=True)

    @staticmethod
    def get_predictions_dir_path(output_dir, create_if_not_exists=True):
        if output_dir is not None:
            predictions_folder = os.path.join(output_dir, "predictions")
        else:
            predictions_folder = "predictions"

        if create_if_not_exists:
            Path(predictions_folder).mkdir(parents=True, exist_ok=True)

        return predictions_folder

    @staticmethod
    def compute_accuracy(pred):
        labels = pred.label_ids.argmax(-1)
        preds = pred.predictions.argmax(-1)
        # calculate accuracy using sklearn's function
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
        }

    @staticmethod
    def compute_classification_metrics(pred):
        labels = pred.label_ids.argmax(-1)
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

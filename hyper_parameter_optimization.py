import logging

import keras_tuner as kt
import yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset_loaders import (
    get_birdclef_datasets,
    get_birdset_dataset,
    get_custom_dataset,
)
from init import init
from misc import hp_audio_and_aug, load_config
from models.binary_cnn_hyper import model_builder
from models.miniresnet_hyper import build_miniresnet_hyper


class DataAwareHyperband(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters

        hp_cfg = hp_audio_and_aug(hp)

        if hp_cfg["data"]["use_dataset"] == "birdset":
            logging.info("[Info] Loading birdset!")
            datasets = get_birdset_dataset(hp_cfg)
        elif hp_cfg["data"]["use_dataset"] == "birdclef":
            logging.info("[Info] Loading Birdclef")
            datasets = get_birdclef_datasets(hp_cfg)
        else:
            logging.info("[Info] Loading custom Dataset")
            datasets = get_custom_dataset(hp_cfg, percentage=0.1)

        early = EarlyStopping(
            monitor="val_recall_at_p90",
            mode="max",
            patience=8,
            min_delta=1e-3,
            restore_best_weights=True,
            verbose=1,
        )

        ret = super().run_trial(
            trial,
            x=datasets["train_ds"],
            validation_data=datasets["val_ds"],
            epochs=hp_cfg["ml"]["epochs_per_fold"],
            callbacks=[early],
            **kwargs,
        )

        del datasets

        return ret


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Load the config
    CONFIG_PATH = "config.yaml"
    logging.info(f"Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)

    if config is None:
        exit(1)

    logging.info(
        f"Running Experiment {config['exp']['name']} for {config['exp']['target']}"
    )

    # Initialize the framework
    init(config["exp"]["random_state"])

    # Train the model
    tuner = DataAwareHyperband(
        hypermodel=build_miniresnet_hyper,
        objective=kt.Objective("val_recall_at_p90", direction="max"),
        max_epochs=100,
        factor=3,
        seed=config["exp"]["random_state"],
        directory="my_dir",
        project_name="miniresnet_final",
    )

    tuner.search()

    print(tuner.results_summary())

    # Get the best trial’s hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    with open("best_hparams.yaml", "w") as f:
        yaml.safe_dump(best_hp.values, f, sort_keys=False)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save("best_model.keras")

    # model = model_builder(best_hps[0])
    #
    # Build the dataset with the best hyperparameters

    # # Retrain the model
    # history = model.fit(
    #     datasets["train_ds"], epochs=50, validation_data=datasets["val_ds"]
    # )
    #
    # val_pr_auc_per_epoch = history.history["val_pr_auc"]
    # best_epoch = val_pr_auc_per_epoch.index(max(val_pr_auc_per_epoch)) + 1
    # print(f"Best epoch: {best_epoch}")
    #
    # hypermodel = tuner.hypermodel.build(best_hps)
    # hypermodel.fit(datasets["train_ds"], epochs=50, validation_data=datasets["val_ds"])
    #
    # eval_result = hypermodel.evaluate(datasets["test_ds"])
    # print(eval_result)

from __future__ import print_function
import os
import datetime
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
import wandb
import argparse
from config.generate_config import load_config_hyper
import models.model_factory as model_factory
import scripts.utils as utils
import optuna
import yaml
import pickle


from trainer_bad import (
    train_one_epoch,
    evaluate,
    load_data,
    create_criterion,
    create_optimizer_and_scheduler,
    final_test,
)


def objective(trial, study_name, config):
    device = torch.device(0)
    best_loss = float("inf")
    best_model_state_dict = None

    wandb.init(
        project=config["wandb_project"],
        group=study_name,
        job_type="optunatrial",
        name=study_name + "_trial:" + format(trial.number, "05d"),
        reinit=True,
    )

    # Suggest parameters
    config["lr"] = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    config["epochs"] = trial.suggest_int("epochs", 10, 100)
    config["batch_size"] = trial.suggest_int("batch_size", 6, 16)
    config["dim"] = trial.suggest_int("dim", 256, 1024, log=True)
    config["depth"] = trial.suggest_int("depth", 2, 5)
    config["heads"] = trial.suggest_int("heads", 4, 8)
    config["mlp_dim"] = trial.suggest_int("mlp_dim", 256, 1024, log=True)
    #config["DS_AUGMENTS_CFG:"][0]["p_prob"] = trial.suggest_float("p_prob", 0.0, 1)
    #config["DS_AUGMENTS_CFG:"][0]["p_max_pts"] = trial.suggest_int(1300, 2000)
    #config["DS_AUGMENTS_CFG:"][4]["p_prob"] = trial.suggest_float(0.0, 1)
    ##config["DS_AUGMENTS_CFG:"][3]["p_prob"] = trial.suggest_float(0.0, 1)
    #config["DS_AUGMENTS_CFG:"][4]["p_stddev"] = trial.suggest_float(0.001, 0.01)
    #config["DS_AUGMENTS_CFG:"][4]["p_clip"] = trial.suggest_float(0.01, 0.1)
    #config["DS_AUGMENTS_CFG:"][5]["p_prob"] = trial.suggest_float(0.0, 1)
    #config["DS_AUGMENTS_CFG:"][6]["p_prob"] = trial.suggest_float(0.0, 1)

    wandb.config.update(config)

    # Data loading code
    print("Loading data from", config["dataset_root"])
    data_loader, data_loader_test, data_loader_val, num_classes = load_data(config)
    print("Number of unique labels (classes):", num_classes)

    model = model_factory.create_model(config, num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Using", torch.cuda.device_count(), "GPUs!")

    model.to(device)

    criterion = create_criterion(config, data_loader, num_classes, device)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, model, data_loader)

    print("Start training")
    start_time = time.time()

    for epoch in range(config["start_epoch"], config["epochs"]):
        train_clip_loss, train_clip_acc = train_one_epoch(
            model, criterion, optimizer, lr_scheduler, data_loader, device, epoch
        )
        (
            val_clip_loss,
            val_clip_acc,
            val_video_acc,
            val_list_video_class_acc,
            val_average_video_class_acc,
            f1,
            confusion_matrix,
        ) = evaluate(model, criterion, data_loader_val, device=device)

        wandb.log(
            {
                "Train Loss": train_clip_loss,
                "Train Clip Acc@1": train_clip_acc,
                "Val Loss": val_clip_loss,
                "Val Clip Acc@1": val_clip_acc,
                "Val Video Acc@1": val_video_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "Val Avg Video Class Acc": val_average_video_class_acc,  # Log the average per-class accuracy for validation videos
                "F1": f1,
            }
        )

        print(
            f"Epoch {epoch} - Train Loss: {train_clip_loss:.4f}, Train Acc@1: {train_clip_acc:.4f}"
        )
        print(
            f"Epoch {epoch} - Validation Loss: {val_clip_loss:.4f}, Validation Acc@1: {val_clip_acc:.4f}, Validation Video Acc@1: {val_video_acc:.4f}"
        )
        print(
            f"Epoch {epoch} - Average Validation Video Class Acc: {val_average_video_class_acc:.4f}"
        )
        print(
            f"Epoch {epoch} - Validation Video Class Acc: {val_list_video_class_acc.tolist()}"
        )
        print(f"Epoch {epoch} - F1: {f1:.4f}")
        print(f"Epoch {epoch} - Confusion Matrix: {confusion_matrix}")

        if val_clip_loss < best_loss:
            best_loss = val_clip_loss
            trial.set_user_attr("best_config", config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    wandb.finish()

    return best_loss


def test_best_model(config, device):
    data_loader, data_loader_test, data_loader_val, num_classes = load_data(config)

    model = model_factory.create_model(config, num_classes)

    model.to(device)

    criterion = create_criterion(config, data_loader, num_classes, device)

    f1, report_str = final_test(
        model,
        criterion,
        data_loader_test,
        device=device,
        output_dir=config["output_dir"],
    )
    print(f1, report_str)


def main(args):
    config = load_config_hyper(args.config)
    utils.set_random_seed(config["seed"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_args"])
    device = torch.device(0)
    study_name = args.config
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(trial, study_name, config), n_trials=config["n_trials"]
    )

    best_trial = study.best_trial
    best_config = best_trial.user_attrs["best_config"]

    with open(os.path.join(config["output_dir"], "best_config.yaml"), "w") as f:
        yaml.dump(best_config, f)

    # Save the study
    with open(os.path.join(config["output_dir"], "study.pkl"), "wb") as f:
        pickle.dump(study, f)

    # Run the final test
    test_best_model(best_config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P4Transformer Model Training")
    parser.add_argument(
        "--config", type=str, default="P4T_BAD2/1", help="Path to the YAML config file"
    )
    args = parser.parse_args()
    main(args)

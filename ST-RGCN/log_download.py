import wandb
import pandas as pd

if __name__ == "__main__":
    wandb.login(key='e5e7b3c0c3fbc088d165766e575853c01d6cb305')

    # Specify your W&B project and run details
    entity = "gvnberaldi"         # Your W&B username or team name
    project = "rgcn-action-classification"   # The W&B project name
    run_id = "sus36aq9"          # The unique ID of the run (you can find this in the URL of your W&B run)

    # Initialize the API
    api = wandb.Api()

    # Fetch the run
    run = api.run(f"{entity}/{project}/{run_id}")

    # Download the metrics as a DataFrame
    history = run.history(keys=["train_accuracy"])  # Replace 'loss' with the metric you want
    window = 10
    history["loss_smooth"] = history["train_accuracy"].rolling(window=window, min_periods=1).mean()

    # Drop the original 'loss' column and rename 'loss_smooth' to 'loss'
    history.drop(columns=["train_accuracy"], inplace=True)
    history.rename(columns={"loss_smooth": "metric"}, inplace=True)
    # Rename the index column to 'step'
    history.reset_index(inplace=True)
    history.rename(columns={"_step": "step"}, inplace=True)
    history.drop(columns=["index"], inplace=True)

    # Ensure 'loss' column is of float type
    history["metric"] = history["metric"].astype(float)

    # Save to CSV with semicolon separator
    history.to_csv("rgcn_train_accuracy.csv", sep=';', index=False)

    print("Metric saved as loss_metric.csv")

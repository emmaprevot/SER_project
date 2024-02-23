from datetime import datetime
from pathlib import Path

import typer
import torch
import git
import json
import pandas as pd

from ser.train import train as run_train
from ser.infer import infer as run_inference
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params
from ser.transforms import transforms, normalize, flip


main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )


@main.command()
def infer(
    exp_name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to load for inference."
    ),
    exp_timestamp: str = typer.Option(
        ..., "-n", "--timestamp", help="Timestamp of experiment to load for inference."
    ),
    label: int = typer.Option(
        6, "-n", "--label", help="Label for inference."
    ),
    flip_img: bool = typer.Option(
        True, "-n", "--flip", help="Flip images."
    ),
):
     """Run the inferences."""
    run_path = RESULTS_DIR / exp_name / exp_timestamp
    # load the model and parameters
    model = torch.load(run_path / "model.pt")
    f = open(run_path / "params.json")
    params = json.load(f)
    f.close()
    
    # Inference !
    
    ts = [normalize]
        
    if flip_img:
        print("Flipping!")
        ts.append(flip)
        
    dataloader = test_dataloader(1, transforms(*ts))
    
    run_inference(
        model,
        params,
        label,
        dataloader
    )
    
    

@main.command()
def print_summary_table():
     """Print a table of all the experiments and their runs."""
    params_list = []
    result_folder = Path(RESULTS_DIR)
    for item in result_folder.iterdir():
        if item.is_dir():
            for sub_item in item.iterdir():
                if sub_item.is_dir():
                    for sub_sub_item in sub_item.iterdir():
                        if sub_sub_item.is_file():
                            if ".json" in str(sub_sub_item):
                                f = open(str(sub_sub_item))
                                params = json.load(f)
                                params_list.append(params)
                                f.close()
    df = pd.DataFrame(params_list)
    print(df)
         

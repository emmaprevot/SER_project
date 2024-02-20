from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import date, datetime
import json

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "trained_models"

import sys
sys.path.append(".")

from ser.data import get_data
from ser.model import Net
from ser.train import train_model

import typer

main = typer.Typer()




@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "-n", "--epoch", help="Number of epochs for training."
    ),
    batch_size: int = typer.Option(
        ..., "-n", "--batch-size", help="Batch size for training."
    ),
    learning_rate: float = typer.Option(
        ..., "-n", "--learning-rate", help="Learning rate for training."
    ),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # epochs = 2
    # batch_size = 1000
    # learning_rate = 0.01

    # save the parameters!

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_dataloader, validation_dataloader = get_data(batch_size)

    # train
    train_losses, val_losses, val_accuracies, trained_model = train_model(epochs, training_dataloader, validation_dataloader, model, optimizer)

    # saving model
    model_path = str(MODEL_DIR) + "/" + name + "_" + str(date.today()) + ".pth"
    torch.save(trained_model.state_dict(), model_path)

    hyp_dict = {
        "name": name,
        "date": str(datetime.now()),
        "epochs": epochs,
        "learning rate": learning_rate,
        "batch size": batch_size
    }

    json_path = str(MODEL_DIR) + "/" + name + "_" + str(date.today()) + ".json"
    with open(json_path, 'w') as f:
        json.dump(hyp_dict, f)

@main.command()
def infer():
    print("This is where the inference code will go")
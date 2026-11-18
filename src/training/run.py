
import json
from pathlib import Path

import torch

from src.config import Config
from src.training.train import (
    create_dataloaders,
    create_model,
    train_model
)


RESUME_FROM = None
# Для продолжения обучения:
# RESUME_FROM = "checkpoints/last_model.pt"


def main():
    torch.manual_seed(Config.SEED)

    train_loader, valid_loader = create_dataloaders()

    model, optimizer, criterion = create_model()

    history = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=Config.EPOCHS,
        checkpoint_path="checkpoints/best_model.pt",
        last_checkpoint_path="checkpoints/last_model.pt",
        resume_from=RESUME_FROM
    )

    history_path = Path("checkpoints/history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()

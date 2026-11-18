import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from src.model.transformer import Transformer
from src.config import Config
from src.data.dataset import TranslationDataset, collate_fn
from pathlib import Path

def create_dataloaders():
    train_data = torch.load(
        "data/tokenized/train.pt",
        map_location="cpu",
        weights_only=True
    )

    valid_data = torch.load(
        "data/tokenized/valid.pt",
        map_location="cpu",
        weights_only=True
    )

    train_dataset = TranslationDataset(
        train_data["src_sequences"],
        train_data["tgt_sequences"]
    )

    valid_dataset = TranslationDataset(
        valid_data["src_sequences"],
        valid_data["tgt_sequences"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, valid_loader

def create_model():
    model = Transformer(
        src_vocab_size=Config.SRC_VOCAB_SIZE,
        tgt_vocab_size=Config.TGT_VOCAB_SIZE,
        d_model=Config.D_MODEL,
        num_heads=Config.NUM_HEADS,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        d_ff=Config.D_FF,
        max_seq_len=Config.MAX_SEQ_LEN,
        dropout=Config.DROPOUT,
        src_padding_idx=Config.PAD_ID,
        tgt_padding_idx=Config.PAD_ID
    )

    model = model.to(Config.DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=Config.PAD_ID
    )

    return model, optimizer, criterion

def train_step(model, optimizer, criterion, src, tgt):
    model.train()

    src = src.to(Config.DEVICE)
    tgt = tgt.to(Config.DEVICE)

    # Decoder получает последовательность без последнего токена
    tgt_input = tgt[:, :-1]

    # Целевые значения — та же последовательность без первого токена
    tgt_labels = tgt[:, 1:]

    optimizer.zero_grad()

    logits = model(src, tgt_input)

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        tgt_labels.reshape(-1)
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        Config.GRAD_CLIP
    )

    optimizer.step()

    return loss.item()

def validation_step(model, criterion, src, tgt):
    model.eval()

    src = src.to(Config.DEVICE)
    tgt = tgt.to(Config.DEVICE)

    tgt_input = tgt[:, :-1]
    tgt_labels = tgt[:, 1:]

    with torch.no_grad():
        logits = model(src, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_labels.reshape(-1)
        )

    return loss.item()

def train_epoch(model, optimizer, criterion, train_loader):
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in train_loader:
        loss = train_step(
            model, optimizer, criterion, src, tgt
        )

        num_tokens = (tgt[:, 1:] != Config.PAD_ID).sum().item()

        total_loss += loss * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens


def validate_epoch(model, criterion, valid_loader):
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in valid_loader:
        loss = validation_step(
            model, criterion, src, tgt
        )

        num_tokens = (tgt[:, 1:] != Config.PAD_ID).sum().item()

        total_loss += loss * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens


def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    valid_loader,
    epochs=Config.EPOCHS,
    checkpoint_path="checkpoints/best_model.pt",
    resume_from=None,
    last_checkpoint_path="checkpoints/last_model.pt"
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = Path(last_checkpoint_path)
    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_valid_loss = float("inf")
    start_epoch = 0

    history = {
        "train_loss": [],
        "valid_loss": []
    }

    # Восстанавливаем предыдущее состояние обучения
    if resume_from is not None:
        checkpoint = torch.load(
            resume_from,
            map_location=Config.DEVICE,
            weights_only=True
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_valid_loss = checkpoint["best_valid_loss"]
        history = checkpoint["history"]

        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(
            model, optimizer, criterion, train_loader
        )

        valid_loss = validate_epoch(
            model, criterion, valid_loader
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f}"
        )

        # Сохраняем лучшую модель
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "best_valid_loss": best_valid_loss,
                    "history": history
                },
                checkpoint_path
            )

            print("Best model saved!")

        # Сохраняем последнее состояние после каждой эпохи
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "best_valid_loss": best_valid_loss,
                "history": history
            },
            last_checkpoint_path
        )

    return history



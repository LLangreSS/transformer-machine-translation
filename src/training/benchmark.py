import time

from src.training.train import (
    create_dataloaders,
    create_model,
    train_step
)


def main():
    train_loader, _ = create_dataloaders()
    model, optimizer, criterion = create_model()

    model.train()

    start_time = time.perf_counter()

    for step, (src, tgt) in enumerate(train_loader):
        train_step(model, optimizer, criterion, src, tgt)

        if step == 9:
            break

    elapsed = time.perf_counter() - start_time

    print(f"10 batches time: {elapsed:.2f} sec")
    print(f"Average batch time: {elapsed / 10:.2f} sec")
    print(
        f"Estimated epoch time: "
        f"{elapsed / 10 * len(train_loader) / 60:.2f} min"
    )


if __name__ == "__main__":
    main()
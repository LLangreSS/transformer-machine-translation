import unittest
import torch

from src.training.train import (
    create_model,
    create_dataloaders,
    train_step
)


class TestTrainingStep(unittest.TestCase):

    def test_full_training_step(self):
        model, optimizer, criterion = create_model()
        train_loader, _ = create_dataloaders()

        src, tgt = next(iter(train_loader))

        initial_weight = (
            model.output_projection.weight.detach().clone()
        )

        loss = train_step(
            model, optimizer, criterion, src, tgt
        )

        updated_weight = model.output_projection.weight.detach()

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreater(loss, 0)
        self.assertFalse(torch.equal(initial_weight, updated_weight))


if __name__ == "__main__":
    unittest.main()
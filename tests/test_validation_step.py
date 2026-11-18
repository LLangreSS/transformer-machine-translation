import unittest
import torch

from src.training.train import (
    create_model,
    create_dataloaders,
    validation_step
)


class TestValidationStep(unittest.TestCase):

    def test_validation_returns_finite_loss(self):
        model, _, criterion = create_model()
        _, valid_loader = create_dataloaders()

        src, tgt = next(iter(valid_loader))

        loss = validation_step(
            model, criterion, src, tgt
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreater(loss, 0)

    def test_validation_does_not_update_weights(self):
        model, _, criterion = create_model()
        _, valid_loader = create_dataloaders()

        src, tgt = next(iter(valid_loader))

        initial_weights = [
            p.detach().clone()
            for p in model.parameters()
        ]

        validation_step(model, criterion, src, tgt)

        for initial, current in zip(initial_weights, model.parameters()):
            self.assertTrue(
                torch.equal(initial, current.detach())
            )

    def test_model_is_in_eval_mode(self):
        model, _, criterion = create_model()
        _, valid_loader = create_dataloaders()

        src, tgt = next(iter(valid_loader))

        validation_step(model, criterion, src, tgt)

        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
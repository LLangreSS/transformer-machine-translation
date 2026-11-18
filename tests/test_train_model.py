import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from src.training.train import train_model


class TestTrainModel(unittest.TestCase):

    def test_training_history(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best_model.pt"

            with patch(
                "src.training.train.train_epoch",
                side_effect=[2.0, 1.5, 1.2]
            ), patch(
                "src.training.train.validate_epoch",
                side_effect=[2.5, 1.8, 2.0]
            ):
                history = train_model(
                    model,
                    optimizer,
                    None,
                    None,
                    None,
                    epochs=3,
                    checkpoint_path=checkpoint
                )

            self.assertEqual(len(history["train_loss"]), 3)
            self.assertEqual(len(history["valid_loss"]), 3)

            self.assertEqual(
                history["valid_loss"],
                [2.5, 1.8, 2.0]
            )

            self.assertTrue(checkpoint.exists())

            saved = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True
            )

            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(saved["valid_loss"], 1.8)


if __name__ == "__main__":
    unittest.main()
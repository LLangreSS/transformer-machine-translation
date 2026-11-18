import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from src.training.train import train_model


class TestResumeTraining(unittest.TestCase):

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best_model.pt"

            # Первые две эпохи
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters())

            with patch(
                "src.training.train.train_epoch",
                side_effect=[2.0, 1.5]
            ), patch(
                "src.training.train.validate_epoch",
                side_effect=[2.5, 1.8]
            ):
                train_model(
                    model,
                    optimizer,
                    None,
                    None,
                    None,
                    epochs=2,
                    checkpoint_path=checkpoint_path
                )

            # Создаём новую модель и восстанавливаем обучение
            resumed_model = torch.nn.Linear(4, 2)
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.parameters()
            )

            with patch(
                "src.training.train.train_epoch",
                return_value=1.2
            ), patch(
                "src.training.train.validate_epoch",
                return_value=1.4
            ):
                history = train_model(
                    resumed_model,
                    resumed_optimizer,
                    None,
                    None,
                    None,
                    epochs=3,
                    checkpoint_path=checkpoint_path,
                    resume_from=checkpoint_path
                )

            self.assertEqual(len(history["train_loss"]), 3)
            self.assertEqual(len(history["valid_loss"]), 3)
            self.assertEqual(history["valid_loss"], [2.5, 1.8, 1.4])

            saved = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True
            )

            self.assertEqual(saved["epoch"], 3)
            self.assertEqual(saved["valid_loss"], 1.4)


if __name__ == "__main__":
    unittest.main()
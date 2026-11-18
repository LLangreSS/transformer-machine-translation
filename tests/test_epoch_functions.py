import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from src.training.train import train_epoch, validate_epoch


class TestEpochFunctions(unittest.TestCase):

    def setUp(self):
        self.batches = [
            (
                torch.tensor([[2, 5, 3, 0]]),
                torch.tensor([[2, 7, 3, 0]])
            ),
            (
                torch.tensor([[2, 8, 9, 3]]),
                torch.tensor([[2, 6, 7, 3]])
            )
        ]

        self.loader = self.batches

    @patch("src.training.train.train_step")
    def test_train_epoch_weighted_loss(self, mock_train_step):
        mock_train_step.side_effect = [2.0, 4.0]

        result = train_epoch(
            None, None, None, self.loader
        )

        # Первый батч: 2 токена, второй: 3 токена.
        expected = (2.0 * 2 + 4.0 * 3) / 5

        self.assertAlmostEqual(result, expected)

    @patch("src.training.train.validation_step")
    def test_validate_epoch_weighted_loss(self, mock_validation_step):
        mock_validation_step.side_effect = [2.0, 4.0]

        result = validate_epoch(
            None, None, self.loader
        )

        expected = (2.0 * 2 + 4.0 * 3) / 5

        self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
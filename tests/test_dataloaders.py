import unittest
import torch

from src.training.train import create_dataloaders
from src.config import Config


class TestDataLoaders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_loader, cls.valid_loader = create_dataloaders()

    def test_loaders_are_not_empty(self):
        self.assertGreater(len(self.train_loader), 0)
        self.assertGreater(len(self.valid_loader), 0)

    def test_batch_shapes(self):
        src, tgt = next(iter(self.train_loader))

        self.assertEqual(src.shape[0], Config.BATCH_SIZE)
        self.assertEqual(tgt.shape[0], Config.BATCH_SIZE)

    def test_batch_dtype(self):
        src, tgt = next(iter(self.train_loader))

        self.assertEqual(src.dtype, torch.long)
        self.assertEqual(tgt.dtype, torch.long)

    def test_padding(self):
        src, tgt = next(iter(self.train_loader))

        self.assertTrue(
            torch.all(src >= 0).item()
        )
        self.assertTrue(
            torch.all(tgt >= 0).item()
        )


if __name__ == "__main__":
    unittest.main()
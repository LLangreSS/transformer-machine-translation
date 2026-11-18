import unittest
import torch
import torch.nn as nn

from src.training.train import create_model
from src.config import Config


class TestTrainingSetup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.optimizer, cls.criterion = create_model()

    def test_model_device(self):
        parameter = next(self.model.parameters())
        self.assertEqual(parameter.device.type, Config.DEVICE.type)

    def test_model_output_projection(self):
        self.assertEqual(
            self.model.output_projection.out_features,
            Config.TGT_VOCAB_SIZE
        )

    def test_optimizer_has_model_parameters(self):
        model_params = {
            id(p) for p in self.model.parameters()
        }

        optimizer_params = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group["params"]
        }

        self.assertEqual(model_params, optimizer_params)

    def test_criterion(self):
        self.assertIsInstance(self.criterion, nn.CrossEntropyLoss)
        self.assertEqual(
            self.criterion.ignore_index,
            Config.PAD_ID
        )


if __name__ == "__main__":
    unittest.main()
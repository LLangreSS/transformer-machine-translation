import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.training.run import main


class TestRun(unittest.TestCase):

    @patch("src.training.run.create_dataloaders")
    @patch("src.training.run.create_model")
    @patch("src.training.run.train_model")
    def test_main(
        self,
        mock_train,
        mock_create_model,
        mock_create_loaders
    ):
        mock_create_loaders.return_value = ("train", "valid")
        mock_create_model.return_value = ("model", "optimizer", "criterion")
        mock_train.return_value = {
            "train_loss": [2.0],
            "valid_loss": [2.5]
        }

        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.json"

            with patch(
                "src.training.run.Path",
                return_value=history_path
            ):
                main()

            self.assertTrue(history_path.exists())
            mock_train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
import unittest
import torch

from src.data.dataset import collate_fn


class TestCollateFn(unittest.TestCase):

    def test_padding(self):
        batch = [
            (
                torch.tensor([2, 5, 3]),
                torch.tensor([2, 7, 3])
            ),
            (
                torch.tensor([2, 8, 9, 3]),
                torch.tensor([2, 6, 3, 4])
            )
        ]

        src, tgt = collate_fn(batch, padding_idx=0)

        expected_src = torch.tensor([
            [2, 5, 3, 0],
            [2, 8, 9, 3]
        ])

        expected_tgt = torch.tensor([
            [2, 7, 3, 0],
            [2, 6, 3, 4]
        ])

        self.assertTrue(torch.equal(src, expected_src))
        self.assertTrue(torch.equal(tgt, expected_tgt))


if __name__ == "__main__":
    unittest.main()
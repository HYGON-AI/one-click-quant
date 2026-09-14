"""CPU format tests; synthetic tensors do not establish model accuracy."""
import unittest
import torch
from src.hy4.format_contract import pack, unpack, narrow_grid


class FormatTests(unittest.TestCase):
    def test_odd_width_roundtrip(self):
        q = torch.tensor([[-7, -1, 0, 1, 7]], dtype=torch.int8)
        packed = pack(q)
        self.assertEqual(tuple(packed.shape), (1, 3))
        self.assertTrue(torch.equal(unpack(packed, 5), q))

    def test_channel_scales(self):
        w = torch.tensor([[0., 0., 0.], [-7., 0., 7.], [-14., 0., 14.]])
        scale, zero, _ = narrow_grid(w)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(scale.shape), (3, 1))
        torch.testing.assert_close(scale, torch.tensor([[1.], [1.], [2.]]))
        self.assertTrue(torch.all(zero == 7))


if __name__ == '__main__':
    unittest.main()

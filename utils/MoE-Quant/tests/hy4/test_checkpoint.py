import tempfile
import unittest
from pathlib import Path
import torch
from src.hy4.checkpoint_writer import CheckpointWriter
from src.hy4.checkpoint_loader import CheckpointLoader


class CheckpointTests(unittest.TestCase):
    def test_recovery_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checkpoint'
            expected = {'expert': 'quantized', 'norm': 'retained'}
            identity = {'source': 'fixture', 'recipe': 'fixture'}
            w = CheckpointWriter(root, identity, expected, max_shard_bytes=64)
            w.add('expert', {'packed': torch.zeros(3, 4, dtype=torch.uint8),
                            'scale': torch.ones(3, 1)},
                  dict(shape=[3, 7], source_sha256='fixture', recipe_sha256='fixture',
                       algorithm='RTN', coverage=0))
            w.flush()
            w = CheckpointWriter(root, identity, expected, 64)
            self.assertTrue(w.contains('expert'))
            with self.assertRaises(ValueError):
                w.finish()
            w.add('norm', {'weight': torch.ones(7, dtype=torch.bfloat16)},
                  dict(source_sha256='fixture'))
            w.finish()
            loader = CheckpointLoader(root, identity, expected)
            self.assertEqual(tuple(loader.projection('expert')[0]['scale'].shape), (3, 1))
            with self.assertRaises(ValueError):
                CheckpointLoader(root, {'source': 'changed'}, expected)
            shard = next(root.glob('*.safetensors'))
            with shard.open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaises(ValueError):
                CheckpointLoader(root, identity, expected)


if __name__ == '__main__':
    unittest.main()

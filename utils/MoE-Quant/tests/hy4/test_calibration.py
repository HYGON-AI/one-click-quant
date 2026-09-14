import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import prepare_hy4_calibration
from src.hy4.hf_hy4_frozen_data import load


class CalibrationTests(unittest.TestCase):
    def test_frozen_manifest_and_eight_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, texts, out = root / 'model', root / 'texts', root / 'out'
            model.mkdir()
            texts.mkdir()
            (model / 'config.json').write_text('{"vocab_size": 4}')
            (model / 'model.safetensors.index.json').write_text('{"weight_map": {}}')
            (model / 'tokenizer.json').write_text('{}')
            meta = {}
            for domain in ('zh', 'en', 'math', 'code'):
                meta[domain] = dict(dataset='fixture', revision='fixture', split='train', license='fixture')
                (texts / (domain + '.jsonl')).write_text('\n'.join(
                    json.dumps(dict(id=i, text=domain + str(i))) for i in range(8)))
            (texts / 'sources.json').write_text(json.dumps(meta))
            tokenizer = SimpleNamespace(encode=lambda *a, **k: [1, 2, 3])
            fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
            argv = ['prepare', '--model', str(model), '--input', str(texts), '--output', str(out),
                    '--per-domain', '8', '--max-length', '2']
            with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'transformers': fake}):
                prepare_hy4_calibration.main()
            all_ids = []
            for rank in range(8):
                tensors, identity = load(out / 'manifest.json', model, out, rank, 8, 32, 2)
                self.assertEqual(len(tensors), 4)
                self.assertTrue(all(t.tolist() == [[1, 2]] for t in tensors))
                all_ids.extend(identity['sample_ids'])
            self.assertEqual(len(set(all_ids)), 32)
            (out / 'zh.json').write_text('{}')
            with self.assertRaises(ValueError):
                load(out / 'manifest.json', model, out, 0, 8, 32, 2)

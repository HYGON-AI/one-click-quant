"""Numerical and loading-regression tests; run inside the analysis environment."""

import unittest
from types import SimpleNamespace

import torch

from utils.auto_quantize.api import channel_candidate, causal_loss, sequence_sum_loss
from utils.auto_quantize.calibration import batch_calibration
from utils.auto_quantize.loading import decode_mxfp4
from utils.auto_quantize.max_calibration import reuse_max_calibration
from utils.auto_quantize.streaming import attach_source_offload


class OptimizationTests(unittest.TestCase):
    def test_max_reuses_stats_without_materializing_second_candidate(self):
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization import model_calib

        torch.manual_seed(19)
        model = torch.nn.Sequential(torch.nn.Linear(32, 8, bias=False), torch.nn.Linear(8, 4, bias=False))
        source = {name: value.detach().clone() for name, value in model.named_parameters()}
        reads = []

        def get(name, shape, dtype, device, **kwargs):
            reads.append(name)
            return source[name].to(dtype=dtype, device=device).clone()

        decoded = SimpleNamespace(runtime=SimpleNamespace(device="cpu"), get_on_device=get)
        model.to("meta")
        attach_source_offload(model, decoded, "cpu")
        with reuse_max_calibration(model) as stats:
            mtq.quantize(model, channel_candidate(4))
            expected = [source[f"{i}.weight"].abs().amax(dim=1, keepdim=True) for i in range(2)]
            for layer, amax in zip(model, expected):
                torch.testing.assert_close(layer.weight_quantizer.amax, amax)
            first_reads = len(reads)
            mtq.set_quantizer_by_cfg(model, channel_candidate(8)["quant_cfg"])
            model_calib.max_calibrate(model)
            self.assertEqual(len(reads), first_reads)
            self.assertEqual(stats["amax_reused"], 2)
            self.assertEqual(stats["writebacks_skipped"], 2)
            for layer, amax in zip(model, expected):
                torch.testing.assert_close(layer.weight_quantizer.amax, amax)
                self.assertTrue(layer._parameters["weight"].is_meta)

    def test_batch_loss_and_gradients_preserve_sequence_sum(self):
        torch.manual_seed(13)
        sequences = [torch.randint(1, 17, (1, length)) for length in (5, 8, 3)]
        batches = batch_calibration(sequences, 2)
        self.assertEqual([b["input_ids"].shape[0] for b in batches], [2, 1])
        logits = torch.randn(2, 8, 17, requires_grad=True)
        loss = sequence_sum_loss(logits, batches[0])
        reference = sum(causal_loss(logits[i:i + 1, :seq.shape[1]], seq) for i, seq in enumerate(sequences[:2]))
        torch.testing.assert_close(loss, reference)
        actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(reference, logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU decode requires CUDA/HIP")
    def test_gpu_decode_all_codes_scales_and_rounding(self):
        from utils.auto_quantize.mxfp4_gpu import decode_mxfp4_gpu

        codes = torch.arange(32, dtype=torch.uint8) % 16
        packed = (codes[::2] | (codes[1::2] << 4)).repeat(255, 1)
        scales = torch.arange(255, dtype=torch.uint8).reshape(255, 1)
        reference = decode_mxfp4(packed, scales)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            actual, invalid = decode_mxfp4_gpu(packed.cuda(), scales.cuda(), dtype)
            expected = reference.to(dtype)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            self.assertEqual(bool(invalid.item()), not bool(torch.isfinite(expected).all()))
        scales[0, 0] = 255
        _, invalid = decode_mxfp4_gpu(packed.cuda(), scales.cuda(), torch.bfloat16)
        self.assertEqual(invalid.item(), 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

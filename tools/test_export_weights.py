import unittest

import torch

from export_weights import required_specs, select_and_validate


class ExportWeightsTests(unittest.TestCase):
    def test_contract_has_exact_greedy_tensor_count(self):
        specs = required_specs()
        self.assertEqual(len(specs), 296)
        self.assertEqual(specs["gpt.h.0.attn.c_attn.weight"], (1280, 3840))
        self.assertNotIn("wte.weight", specs)
        self.assertNotIn("wpe.weight", specs)

    def test_validation_rejects_missing_tensor(self):
        with self.assertRaisesRegex(ValueError, "missing: mel_embedding.weight"):
            select_and_validate({})

    def test_validation_rejects_wrong_shape(self):
        state = {
            name: torch.empty((1,), dtype=torch.float32)
            for name in required_specs()
        }
        with self.assertRaisesRegex(ValueError, "shape: mel_embedding.weight"):
            select_and_validate(state)


if __name__ == "__main__":
    unittest.main()

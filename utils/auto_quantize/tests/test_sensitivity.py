"""Report truncation must not truncate decisions or hide invalid tail groups."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from utils.auto_quantize.sensitivity import (
    format_sensitivity_report, save_sensitivity_json, save_sensitivity_report, sensitivity_ranking,
)


def make_plan(count):
    groups, bits = {}, {}
    for i in range(count):
        name = f"layers.{i}.linear"
        bits[name] = 4
        groups[name] = {"status": "searched", "selected_bits": 4, "module_names": [name],
                        "scores": {"4": float(count - i), "8": 0.5}, "weight_elements": 32,
                        "allowed_bits": [4, 8], "unobserved_modules": [name] if i == 0 else []}
    return {"model": {"model_id": "report-test"}, "method": "modelopt_gradient", "groups": groups,
            "module_bits": bits, "effective_bits_target": 4.2, "actual_effective_bits": 4.0}


class SensitivityReportTests(unittest.TestCase):
    def test_both_reports_keep_ten_and_leave_plan_complete(self):
        plan = make_plan(12)
        original = copy.deepcopy(plan)
        artifacts = Path(__file__).parent / ".artifacts"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as directory:
            txt = Path(directory) / "plan.sensitivity.txt"
            js = Path(directory) / "plan.sensitivity.json"
            save_sensitivity_report(plan, txt)
            save_sensitivity_json(plan, js)
            report = txt.read_text()
            payload = json.loads(js.read_text())
            self.assertIn("Showing top 10 of 12", report)
            self.assertIn("layers.9.linear", report)
            self.assertNotIn("layers.10.linear", report)
            self.assertNotIn("Group membership (", report)
            self.assertIn("字段说明", report)
            self.assertIn("unobserved=1", report)
            self.assertEqual(payload["reported_groups"], 10)
            self.assertEqual(payload["total_groups"], 12)
            self.assertTrue(payload["truncated"])
            self.assertEqual([r["rank"] for r in payload["ranking"]], list(range(1, 11)))
            self.assertTrue(all("module_names" not in row for row in payload["ranking"]))
            self.assertIn("int4_score", payload["field_descriptions"])
            with self.assertRaises(FileExistsError):
                save_sensitivity_report(plan, txt)
        self.assertEqual(plan, original)
        self.assertEqual(len(sensitivity_ranking(plan)), 12)

    def test_small_plan_and_requested_preview(self):
        self.assertIn("Showing top 2 of 2", format_sensitivity_report(make_plan(2)))
        self.assertIn("Showing top 3 of 12", format_sensitivity_report(make_plan(12), top_k=3))
        self.assertIn("Showing top 10 of 12", format_sensitivity_report(make_plan(12), top_k=100))
        self.assertIn("Showing top 10 of 12", format_sensitivity_report(make_plan(12), top_k=None))

    def test_invalid_omitted_group_still_fails_validation(self):
        plan = make_plan(12)
        plan["groups"]["layers.11.linear"]["scores"]["4"] = -1
        with self.assertRaisesRegex(ValueError, "Invalid sensitivity"):
            format_sensitivity_report(plan)


if __name__ == "__main__":
    unittest.main()

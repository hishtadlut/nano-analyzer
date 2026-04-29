import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ai_act_poc


class AiActPocTests(unittest.TestCase):
    def test_biometric_prefilter_matches_real_terms_not_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "app").mkdir()
            (repo / "app" / "safe.py").write_text(
                "class PaymentInterface:\n    pass\n",
                encoding="utf-8",
            )
            (repo / "app" / "risk.py").write_text(
                "import face_recognition\nresult = face_recognition.compare_faces([], image)\n",
                encoding="utf-8",
            )

            signals = ai_act_poc.find_biometric_signals(repo)

            self.assertEqual([item["file"] for item in signals], ["app/risk.py"])
            self.assertEqual(signals[0]["matches"][0]["term"], "face")

    def test_report_generation_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            report = ai_act_poc.build_report(
                repo,
                [
                    {
                        "file": "app/face_login.py",
                        "matches": [{"line": 4, "term": "OpenCV", "snippet": "cv2.CascadeClassifier(...)"}],
                    }
                ],
            )

            paths = ai_act_poc.generate_compliance_files(
                repo,
                report,
                plan_markdown="# Plan\n",
                pr_body="# PR\n",
            )

            data = json.loads(paths["json_report"].read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "possible_trigger_found")
            self.assertIn("Fundamental rights impact assessment", paths["required_actions"].read_text(encoding="utf-8"))
            self.assertTrue(paths["plan"].exists())
            self.assertTrue(paths["pr_body"].exists())

    def test_opencode_command_construction(self):
        command = ai_act_poc.build_opencode_command("deepseek/deepseek-v4-pro", "plan this")
        self.assertEqual(command, ["opencode", "run", "--model", "deepseek/deepseek-v4-pro", "plan this"])

        dangerous = ai_act_poc.build_opencode_command(
            "deepseek/deepseek-v4-pro",
            "implement this",
            dangerously_skip_permissions=True,
        )
        self.assertEqual(
            dangerous,
            [
                "opencode",
                "run",
                "--model",
                "deepseek/deepseek-v4-pro",
                "--dangerously-skip-permissions",
                "implement this",
            ],
        )

    def test_changed_file_detection_supports_added_and_modified_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
            (repo / "README.md").write_text("hello\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

            (repo / "README.md").write_text("hello again\n", encoding="utf-8")
            (repo / "new.py").write_text("print('new')\n", encoding="utf-8")

            self.assertEqual(ai_act_poc.get_changed_files(repo), ["README.md", "new.py"])

            (repo / "README.md").unlink()
            with self.assertRaisesRegex(RuntimeError, "Only added and modified"):
                ai_act_poc.get_changed_files(repo)

    def test_github_payload_helpers(self):
        blob_payload = ai_act_poc.make_github_blob_payload(b"hello")
        self.assertEqual(blob_payload, {"content": "aGVsbG8=", "encoding": "base64"})

        tree_entry = ai_act_poc.make_tree_entry("docs\\risk.md", "abc123")
        self.assertEqual(
            tree_entry,
            {"path": "docs/risk.md", "mode": "100644", "type": "blob", "sha": "abc123"},
        )

        pr_payload = ai_act_poc.make_pr_payload("Title", "branch", "main", "Body")
        self.assertTrue(pr_payload["draft"])
        self.assertEqual(pr_payload["head"], "branch")


if __name__ == "__main__":
    unittest.main()

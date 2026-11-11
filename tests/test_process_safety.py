"""Synthetic process tests for timeout and subprocess failure handling."""

import errno
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import score_integrated as integrated
import score_pronunciation as pronunciation
import process_safety
from process_safety import run_bounded


class ProcessSafetyTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_timeout_terminates_and_reaps_descendant_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            registry = Path(directory) / "groups.txt"
            script = (
                "import os, pathlib, subprocess, sys, time, process_safety; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
                "process_safety._update_registry(os.environ[process_safety.PROCESS_GROUP_REGISTRY_ENV],child.pid); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
            )
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded(
                    [sys.executable, "-c", script, str(pid_file)], timeout=0.25,
                    capture_output=True, text=True,
                    env={**os.environ, process_safety.PROCESS_GROUP_REGISTRY_ENV: str(registry)},
                    cleanup_process_groups=True,
                )
            self.assertLess(time.monotonic() - started, 4.0)
            child_pid = int(pid_file.read_text())
            for _ in range(100):
                try:
                    os.kill(child_pid, 0)
                except OSError as error:
                    if error.errno == errno.ESRCH:
                        break
                    raise
                time.sleep(0.02)
            else:
                self.fail(f"descendant process {child_pid} survived its parent timeout")

    def test_missing_executable_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            run_bounded(["/missing/pte-scoring-command"], timeout=1, capture_output=True, text=True)

    def test_cancellation_terminates_the_child_group(self):
        process = Mock()
        process.pid = 123
        process.communicate.side_effect = [KeyboardInterrupt, ("", "")]
        with patch.object(process_safety.subprocess, "Popen", return_value=process), patch.object(
            process_safety, "_terminate_process_group",
        ) as terminate:
            with self.assertRaises(KeyboardInterrupt):
                run_bounded(["synthetic"], timeout=1, capture_output=True, text=True)
        terminate.assert_called_once_with(process)

    def test_pronunciation_timeout_is_structured(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(
            integrated, "run_bounded", side_effect=subprocess.TimeoutExpired("scorer", 300),
        ):
            result = integrated._build_pronunciation(Path("fixture.wav"), "test", 1)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"]["status"], "STAGE_TIMEOUT")
        self.assertIn("terminated", result["error"]["message"])

    def test_kaldi_stage_timeout_reports_the_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(pronunciation.ScoreError) as raised:
                pronunciation.run_stage(
                    "synthetic_hang", [sys.executable, "-c", "import time; time.sleep(60)"],
                    root, {}, root / "logs", timeout=0.1,
                )
        self.assertEqual(raised.exception.status, "STAGE_TIMEOUT")
        self.assertEqual(raised.exception.stage, "synthetic_hang")

    def test_malformed_pronunciation_output_is_reported(self):
        completed = subprocess.CompletedProcess(["scorer"], 0, "not-json", "")
        with patch.object(Path, "is_file", return_value=True), patch.object(
            integrated, "run_bounded", return_value=completed,
        ):
            result = integrated._build_pronunciation(Path("fixture.wav"), "test", 1)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"]["status"], "PRONUNCIATION_FAILED")


if __name__ == "__main__":
    unittest.main()

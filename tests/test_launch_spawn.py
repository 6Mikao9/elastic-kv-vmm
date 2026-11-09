"""Exercise the package launcher through a real multiprocessing spawn."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LaunchSpawnTest(unittest.TestCase):
    def test_package_launcher_installs_minimax_adapter_in_spawned_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            stub = Path(temporary)
            (stub / "sitecustomize.py").write_text(
                """\
import os
import adaptive_sparse.minimax_runtime as runtime

def install():
    os.environ["TEST_MINIMAX_HOOK_PID"] = str(os.getpid())
    return ("stub_sm80",)

runtime.install_minimax_runtime = install
""",
                encoding="utf-8",
            )
            (stub / "sglang/srt/entrypoints").mkdir(parents=True)
            (stub / "sglang/__init__.py").touch()
            (stub / "sglang/srt/server_args.py").write_text(
                "def prepare_server_args(argv): return argv\n", encoding="utf-8"
            )
            (stub / "sglang/srt/entrypoints/http_server.py").write_text(
                """\
import multiprocessing
import os

def check_worker_adapter():
    assert os.environ.get("TEST_MINIMAX_HOOK_PID") == str(os.getpid()), (
        "MiniMax adapter did not install in the spawned worker"
    )

def launch_server(args):
    assert os.environ.get("TEST_MINIMAX_HOOK_PID") == str(os.getpid())
    worker = multiprocessing.get_context("spawn").Process(target=check_worker_adapter)
    worker.start()
    worker.join(10)
    if worker.is_alive():
        worker.kill()
        worker.join()
        raise AssertionError("Stub worker timed out")
    assert worker.exitcode == 0, worker.exitcode
""",
                encoding="utf-8",
            )
            env = dict(os.environ, PYTHONPATH=os.pathsep.join((temporary, str(ROOT))),
                       PYTHONDONTWRITEBYTECODE="1")
            env.pop("_ELASTIC_KV_INSTALL_ON_IMPORT", None)
            for module in ("elastic_kv", "elastic_kv.launch"):
                with self.subTest(module=module):
                    result = subprocess.run(
                        [sys.executable, "-m", module,
                         "--elastic-runtime-profile", "minimax-hisparse"],
                        env=env, cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=20,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from elastic_kv.build_vmm import DEFAULT_SOURCE, build_command, main


class BuildVmmTest(unittest.TestCase):
    def test_source_is_packaged_next_to_runtime_binary(self):
        self.assertTrue(DEFAULT_SOURCE.is_file())
        self.assertEqual(DEFAULT_SOURCE.parent.name, "native")

    def test_build_command_uses_requested_toolchain(self):
        source = Path("/pkg/native/elastic_vmm.cpp")
        output = Path("/tmp/libelastic_vmm.so")

        command = build_command(
            source,
            output,
            cxx="g++-13",
            cuda_home=Path("/opt/cuda"),
        )

        self.assertEqual(command[0], "g++-13")
        self.assertIn("-shared", command)
        self.assertEqual(
            Path(command[command.index("-I") + 1]),
            Path("/opt/cuda/include"),
        )
        self.assertEqual(command[-2:], ["-o", str(output)])

    def test_dry_run_does_not_require_a_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "elastic_vmm.cpp"
            source.write_text("// test\n", encoding="utf-8")
            include = root / "cuda" / "include"
            include.mkdir(parents=True)
            (include / "cuda.h").write_text("// test\n", encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                status = main(
                    [
                        "--source",
                        str(source),
                        "--output",
                        str(root / "libelastic_vmm.so"),
                        "--cuda-home",
                        str(root / "cuda"),
                        "--cxx",
                        "test-cxx",
                        "--dry-run",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertTrue(output.getvalue().startswith("test-cxx "))

    def test_build_invokes_compiler_and_reports_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "elastic_vmm.cpp"
            source.write_text("// test\n", encoding="utf-8")
            include = root / "cuda" / "include"
            include.mkdir(parents=True)
            (include / "cuda.h").write_text("// test\n", encoding="utf-8")
            output = root / "libelastic_vmm.so"

            def produce(command, **_kwargs):
                Path(command[-1]).write_bytes(b"native")

            with patch(
                "elastic_kv.build_vmm.subprocess.run", side_effect=produce
            ) as run:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    status = main(
                        [
                            "--source",
                            str(source),
                            "--output",
                            str(output),
                            "--cuda-home",
                            str(root / "cuda"),
                        ]
                    )

        self.assertEqual(status, 0)
        run.assert_called_once()
        self.assertEqual(stdout.getvalue().strip(), str(output))


if __name__ == "__main__":
    unittest.main()

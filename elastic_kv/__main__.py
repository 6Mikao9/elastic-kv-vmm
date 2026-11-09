"""Allow ``python -m elastic_kv`` to launch the inference server."""

from runpy import run_module


if __name__ == "__main__":
    # multiprocessing spawn skips package __main__ modules. Expose the actual
    # launch module as __main__ so workers rerun its import-time adapter hook.
    run_module("elastic_kv.launch", run_name="__main__", alter_sys=True)

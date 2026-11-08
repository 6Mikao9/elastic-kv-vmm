"""Allow ``python -m elastic_kv`` to launch the inference server."""

from .launch import main


if __name__ == "__main__":
    raise SystemExit(main())

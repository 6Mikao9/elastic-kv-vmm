"""Spawn-safe launch entry point for the pinned SGLang adapter."""
from .adapter import install

install()

if __name__ == "__main__":
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.entrypoints.http_server import launch_server
    import sys
    launch_server(prepare_server_args(sys.argv[1:]))

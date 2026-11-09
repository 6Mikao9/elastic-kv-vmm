"""Legacy MiniMax launcher; use ``elastic-sglang`` for new integrations."""
import os
import sys

from adaptive_sparse.minimax_runtime import install_minimax_runtime

install_minimax_runtime()

if __name__ == "__main__":
    from sglang.launch_server import (
        kill_process_tree,
        load_plugins,
        prepare_server_args,
        run_server,
    )

    load_plugins()
    args = prepare_server_args(sys.argv[1:])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

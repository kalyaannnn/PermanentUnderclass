"""Generate bindings reproducibly from the locked grpcio-tools environment."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare without rewriting")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    relative = Path("lagrl/sandbox/proto/sandbox.proto")
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) if args.check else root / "src"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{root / 'proto'}",
                f"--python_out={output}",
                f"--grpc_python_out={output}",
                str(relative),
            ],
            cwd=root / "proto",
            check=True,
        )
        if args.check:
            for filename in ("sandbox_pb2.py", "sandbox_pb2_grpc.py"):
                path = relative.parent / filename
                if (output / path).read_bytes() != (root / "src" / path).read_bytes():
                    raise SystemExit(f"generated binding differs: {path}")


if __name__ == "__main__":
    main()

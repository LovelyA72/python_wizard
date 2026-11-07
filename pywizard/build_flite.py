"""Build the parameter-only Flite helper and its registered English voices."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing Flite source checkout (otherwise downloaded)")
    parser.add_argument("--build-dir", type=Path, help="Keep intermediate build files here")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "bin")
    args = parser.parse_args()
    if not shutil.which("cmake"):
        parser.error("CMake is required to build Flite.")
    with tempfile.TemporaryDirectory(prefix="python-wizard-flite-") as temporary:
        build = (args.build_dir or Path(temporary)).resolve()
        configure = ["cmake", "-S", str(Path(__file__).parent / "native"), "-B", str(build),
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_PREFIX=" + str(args.output.resolve())]
        if args.source:
            configure.append("-DFLITE_SOURCE=" + str(args.source.resolve()))
        if os.name == "nt" and shutil.which("gcc") and shutil.which("mingw32-make"):
            configure.extend(["-G", "MinGW Makefiles"])
        subprocess.run(configure, check=True)
        subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--parallel", "4"], check=True)
        subprocess.run(["cmake", "--install", str(build), "--config", "Release"], check=True)


if __name__ == "__main__":
    main()

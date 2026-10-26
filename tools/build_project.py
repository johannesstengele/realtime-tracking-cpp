import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Provision pinned dependencies, build and run all tests")
    parser.add_argument("--preset", choices=("dev", "release"), default="dev")
    parser.add_argument("--vcpkg-root", type=Path)
    parser.add_argument("--triplet")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "vcpkg.json").read_text(encoding="utf-8"))
    baseline = manifest["builtin-baseline"]
    dependency_root = (arguments.vcpkg_root or root / "out" / "dependencies" / "vcpkg").resolve()
    machine = platform.machine().lower()
    if not arguments.triplet and machine not in ("amd64", "x86_64"):
        parser.error("Select an explicit vcpkg triplet for this architecture")
    triplet = arguments.triplet or ("x64-windows" if os.name == "nt" else "x64-linux")
    try:
        if not dependency_root.exists():
            dependency_root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "https://github.com/microsoft/vcpkg.git", str(dependency_root)], check=True)
            subprocess.run(["git", "-C", str(dependency_root), "checkout", baseline], check=True)
        revision = subprocess.run(["git", "-C", str(dependency_root), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        if revision != baseline:
            raise ValueError("The dependency manager must be at the manifest baseline; existing checkouts are preserved")
        bootstrap = dependency_root / ("bootstrap-vcpkg.bat" if os.name == "nt" else "bootstrap-vcpkg.sh")
        subprocess.run([str(bootstrap), "-disableMetrics"], check=True)
        build = root / "out" / "managed" / arguments.preset
        configuration = "Debug" if arguments.preset == "dev" else "Release"
        subprocess.run(["cmake", "--preset", arguments.preset, "-B", str(build),
                        "-DCMAKE_TOOLCHAIN_FILE=" + str(dependency_root / "scripts" / "buildsystems" / "vcpkg.cmake"),
                        "-DVCPKG_TARGET_TRIPLET=" + triplet, "-DVCPKG_MANIFEST_FEATURES=tests",
                        "-DPython3_EXECUTABLE=" + sys.executable], cwd=root, check=True)
        subprocess.run(["cmake", "--build", str(build), "--config", configuration, "--parallel", "4"], check=True)
        subprocess.run(["ctest", "--test-dir", str(build), "-C", configuration,
                        "--output-on-failure", "--no-tests=error"], check=True)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

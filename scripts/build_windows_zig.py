"""Optional project-local Windows build when MSVC is unavailable.

Install pybind11 and ziglang==0.13.0 into the selected Python environment first.
Normal wheel builds continue to use the CMake build backend.
"""
from pathlib import Path
import subprocess
import sys

import pybind11
import ziglang


def main():
    if sys.platform != 'win32':
        raise SystemExit('This fallback targets Windows only')
    root=Path(__file__).resolve().parents[1]
    base=Path(sys.base_prefix)
    library=base/'libs'/f'python{sys.version_info.major}{sys.version_info.minor}.lib'
    if not library.exists():
        raise SystemExit(f'Python development library missing: {library}')
    subprocess.run([
        str(Path(ziglang.__file__).parent/'zig.exe'),'c++','-shared','-O2','-std=c++20',
        '-target','x86_64-windows-gnu','-I',pybind11.get_include(),'-I',str(base/'include'),
        str(root/'cpp/scheduler.cpp'),str(library),'-o',str(root/'src/gpu_runtime/_scheduler.pyd')
    ],check=True,cwd=root)


if __name__=='__main__':main()

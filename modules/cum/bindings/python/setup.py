import os
from pathlib import Path

from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
CUM_INCLUDE_DIR = Path(os.environ.get("CUM_INCLUDE_DIR", ROOT.parent / "C" / "include"))
CUM_LIBRARY_DIR = os.environ.get("CUM_LIBRARY_DIR")
CUM_LIBRARIES = os.environ.get("CUM_LIBRARIES", "cum_C").split()

extension = Extension(
    "cumpy._core",
    sources=[str(ROOT / "src" / "cumpy" / "module.c")],
    include_dirs=[str(CUM_INCLUDE_DIR)],
    library_dirs=[CUM_LIBRARY_DIR] if CUM_LIBRARY_DIR else [],
    libraries=CUM_LIBRARIES,
)

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[extension],
)

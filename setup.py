import glob
import numpy as np
import os
import sys

from distutils.core import setup
from distutils.extension import Extension
from distutils.util import convert_path
from typing import List, Dict, Any


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


if "--use-cython" in sys.argv:
    USE_CYTHON = True
    sys.argv.remove("--use-cython")
    from Cython.Distutils import build_ext

    import Cython.Compiler.Options

    Cython.Compiler.Options.annotate = True
else:
    USE_CYTHON = False

cmdclass = {}
ext_modules: List[Extension] = []
include_dirs = [np.get_include()]
extension = ".pyx" if USE_CYTHON else ".c"
cython_files = glob.glob("datascope/**/*.pyx")
if USE_CYTHON:
    cmdclass.update({"build_ext": build_ext})


for filepath in cython_files:
    basepath = os.path.splitext(filepath)[0]
    modulename = ".".join(basepath.split(os.sep))
    sourcepath = basepath + extension
    ext_modules.append(Extension(modulename, [sourcepath], include_dirs=include_dirs))

main_ns: Dict[str, Any] = {}
ver_path = convert_path("datascope/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

install_requires = parse_requirements("requirements.txt")
extras_require = {"dev": parse_requirements("requirements-dev.txt"), "exp": parse_requirements("requirements-exp.txt")}
extras_require_all = set(r for e in extras_require.values() for r in e)
extras_require["complete"] = extras_require_all


setup(
    name="datascope",
    version=main_ns["__version__"],
    packages=["datascope", "datascope.algorithms", "datascope.utils", "datascope.inspection", "datascope.importance"],
    ext_modules=ext_modules,
    license="MIT",
    author_email="easeml@ds3lab.com",
    url="https://ease.ml/datascope/",
    description="Measuring data importance over ML pipelines using the Shapley value.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass=cmdclass,
)

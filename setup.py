#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import re
import shutil
from distutils import cmd
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


def version_scheme(version):
    version_str = str(version.tag)
    if version.distance is not None and version.distance > 0:
        version_str += f".dev{version.distance}"
    if version.dirty:
        version_str += "+dirty"
    return version_str


def local_scheme(version):
    return ""


class CleanCommand(cmd.Command):
    """Custom clean command to tidy up the project root."""

    FILES_TO_SEARCH_LIST = ["./build", "./dist", "./*.pyc", "./*.tgz", "./*.egg-info"]

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for files_to_search in self.FILES_TO_SEARCH_LIST:
            files_to_delete = glob(files_to_search)
            for file_to_delete in files_to_delete:
                print(f"Removing {file_to_delete}")
                shutil.rmtree(file_to_delete)


setup(
    name="LovelacePM",
    use_scm_version={
        "local_scheme": "dirty-tag",
        "write_to": "src/LovelacePM/_version.py",
        "fallback_version": "0.0.0",
    },
    description="An example package. Generated with cookiecutter-pylibrary.",
    long_description="%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.md")
        ),
    ),
    author="Guilhem Lavabre",
    author_email="guilhem.lavabre@airseas.com",
    url="https://github.com/glavabreairseas/LovelacePM",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Topic :: Utilities",
    ],
    project_urls={
        "Issue Tracker": "https://github.com/glavabreairseas/LovelacePM/issues",
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy<2,>=1.16.6",
        "scipy>=1.12.0",
        "numpy-quaternion",
        "func-timeout",
        "matplotlib",
        "cloudpickle",
        "LoveUpdate",
        "multiprocess",
    ],
    extras_require={
        "dev": [
            "invoke",
            "pytest",
            "tox",
            "black",
            "flake8",
            "twine",
            "plantuml",
            "py2puml",
            "pytest-cov",
        ]
    },
    setup_requires=["setuptools_scm>=3.3.1", "wheel"],
)

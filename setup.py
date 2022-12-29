import pathlib

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

LIBRARY_NAME = "library"  # Rename according to te "library" folder

# List of requirements
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement) for requirement in parse_requirements(requirements_txt)
    ]

setup(
    name=LIBRARY_NAME,
    packages=find_packages(include=[LIBRARY_NAME]),
    version="0.1.0",
    description="Description",
    author="Author",
    license="MIT",
    install_requires=install_requires,
    setup_requires=["pytest-runner"],
    tests_requires=["pytest==4.4.1"],
    test_suite="tests",
)

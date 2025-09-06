# setup.py

from setuptools import setup, find_packages

setup(
    name="i_drill",
    version="0.1.0",
    # Tells setuptools to search for packages in the src folder
    package_dir={"": "src"},
    # Finds all packages in src (i.e. i_drill)
    packages=find_packages(where="src"),
    author="I-Drill Team",
    description="Intelligent Drilling Rig Automation System.",
    install_requires=open('requirements.txt').read().splitlines(),
)
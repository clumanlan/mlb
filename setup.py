from setuptools import setup, find_packages

setup(
    name="mlbdk",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
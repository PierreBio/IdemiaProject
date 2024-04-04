from setuptools import setup, find_packages

setup(
    name="IdemiaProject",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.7.1",
        "pycocotools==2.0.7",
        "setuptools==60.2.0",
        "scikit-image==0.22.0",
    ]
)

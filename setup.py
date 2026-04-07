from setuptools import setup, find_packages

setup(
    name="quant_toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "matplotlib",
    ],
    python_requires=">=3.8",
)

from setuptools import find_packages, setup

setup(
    name="scholarqa",
    version="0.1.0",
    packages=find_packages(where="api"),
    package_dir={"": "api"},
    install_requires=[
        "pandas",
        "transformers",
        "torch",
        "sentence-transformers",
        "FlagEmbedding",
        "anyascii",
        "langsmith",
    ],
    python_requires=">=3.8",
)

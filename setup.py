from setuptools import setup, find_packages

setup(
    name="JaxOptimus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
        "docs": [
            "sphinx>=4.0.2",
            "sphinx-rtd-theme>=0.5.2",
            "nbsphinx>=0.8.6",
            "jupyter>=1.0.0",
        ],
    },
    python_requires=">=3.8",
)
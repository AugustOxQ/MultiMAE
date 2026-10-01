from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(name: str) -> list[str]:
    lines = (Path(__file__).parent / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="mmae",
    version="1.0.0",
    description="Multimodal masked autoencoder (fusion MAE) on CLIP towers",
    packages=find_packages(include=["mmae", "mmae.*"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": ["pytest>=8"]},
    python_requires=">=3.10",
)

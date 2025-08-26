from setuptools import setup, find_packages
from pathlib import Path


def read_requirements(req_path: str):
    path = Path(__file__).parent / req_path
    if not path.exists():
        return []
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="multi-mae",
    version="0.1.0",
    description="MultiMAE package with MAE model, hooks, metrics and utils",
    packages=find_packages(),  # this will include 'src' and 'src.*' packages
    include_package_data=True,
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.10",
)

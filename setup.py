import setuptools
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf8").read()

base_packages = [
    "scikit-learn>=0.22.1",
    "pandas>=1.2.3",
    "matplotlib>=3.2.1",
    "scipy>=1.4.1",
    "tqdm>=4.41.0",
    "shap>=0.38.1",
    "numpy>=1.20.2",
    "xgboost>=0.82",
    "probatus>=1.5.1",
    "hyperopt>=0.2.4",
    "mlflow>=1.14.0",
    "lightgbm>=2.3.1",
    "seaborn>=0.10.1",
    "slicer>=0.0.7",
    "google-api-python-client>=2.5.0",
]

setuptools.setup(
    name="pharmAutoML",
    version="1.0.0",
    description="",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Gengbo Liu",
    author_email="liug28@gene.com",
    license="Apache License",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Apache License",
        "Operating System :: OS Independent",
    ],
    install_requires=base_packages,
    url="https://github.roche.com/liug28/PharmAutoML-mlflow",
    zip_safe=False,
)
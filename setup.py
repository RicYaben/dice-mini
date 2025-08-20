from setuptools import setup

setup(
    name="dice",
    version="0.1.0",
    description="DICE mini",
    license="MIT",
    author="Ricardo Yaben",
    author_email="rmyl@dtu.dk",
    packages=["dice", "plots"],
    install_requires=[
        "pandas",
        "duckdb",
        "numpy",
        "plotly"
    ]
)
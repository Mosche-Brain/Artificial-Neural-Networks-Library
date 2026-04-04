from setuptools import setup, find_packages

setup(
    name="yann",
    version="0.0.1",
    author="Świadomy Mężczyzna",
    author_email="swiadomy.mezczyzna@proton.me",
    description="Lightweight, modular and efficient library for building machine learning models. Written in C++.",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

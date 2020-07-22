import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python-iqif-twetto",
    version="0.0.2",
    author="twetto",
    author_email="franky85912@gmail.com",
    description="A Python library for the Integer Quadratic Integrate-and-Fire neuron API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/twetto/python-iqif",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)


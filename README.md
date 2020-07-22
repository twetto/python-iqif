# python-iqif
A Python library for the Integer Quadratic Integrate-and-Fire neuron API.

## Dependencies

### Runtime:

* python

* [iq-neuron](https://github.com/twetto/iq-neuron)

### Buildtime:

* python-setuptools

* python-wheel

* base-devel (Arch-based packaging)

* pip (Universal installation)

## Installation

### Universal installation

```bash
pip install .
```

### Arch-based installation

First download the PKGBUILD, goto the working directory, then

```bash
makepkg -si
```

Uninstall the package with `sudo pacman -Rs python-iqif`.


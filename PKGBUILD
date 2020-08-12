# Maintainer: twetto <franky85912@gmail.com>
pkgname=python-iqif
pkgver=0.2.1
pkgrel=1
pkgdesc="A Python library for the Integer Quadratic Integrate-and-Fire neuron API."
arch=('any')
url="https://github.com/twetto/python-iqif"
license=('MIT')
depends=('python' 'iq-neuron')
makedepends=('git' 'python-setuptools') # 'bzr', 'git', 'mercurial' or 'subversion'
source=('git+https://github.com/twetto/python-iqif.git')
sha256sums=('SKIP')

# Please refer to the 'USING VCS SOURCES' section of the PKGBUILD man page for
# a description of each element in the source array.

pkgver() {
	cd "$srcdir/${pkgname}"

# Git, tags available
    printf "%s" "$(git describe --long --tags | sed 's/^v//;s/\([^-]*-g\)/r\1/;s/-/./g')"

}

build() {
	cd "$srcdir/${pkgname}"
    python setup.py build
}

package() {
	cd "$srcdir/${pkgname}"
    python setup.py install --root="$pkgdir" --optimize=1 --skip-build
}

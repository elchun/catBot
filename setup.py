import os

from  setuptools import find_packages, setup
packages = find_packages()
for p in packages:
    assert p.startswith('catbot')

setup(
    name='catBot',
    author='Ethan Chun, Kerlina Liu',
    license='MIT',
    packages=packages,
    package_data={}
)

# https://stackoverflow.com/questions/58193205/custom-python-package-not-found

from setuptools import setup, find_packages

setup(
    name='pyimagematch',
    version='0.1',
    description='python bindings for image matching algorithms.',
    packages=find_packages(exclude=['tests']),
    package_data={'': ['clib/*.so', 'assets/*']},
    include_package_data=True,
    install_requires=[],
)
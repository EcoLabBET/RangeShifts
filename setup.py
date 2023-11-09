from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='RangeShifts',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Sofoklis Mouratidis',
    author_email='s.mouratidis@uoi.gr',
    description='A package for handling range shifts in ecological data.',
    url='https://github.com/EcoLabBET/RangeShifts',
    classifiers=[]
)

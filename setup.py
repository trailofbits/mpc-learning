from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="MPC-learning",
    version="1.0",
    install_requires=requirements,
    packages=['src','examples','data'],
    long_description=open("README.md").read(),
    platforms=['any']
)
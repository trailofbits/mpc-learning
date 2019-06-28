from setuptools import setup

setup(
    name="MPC-learning",
    version="1.0",
    packages=['src','examples','data'],
    long_description=open("README.md").read(),
    platforms=['any']
)
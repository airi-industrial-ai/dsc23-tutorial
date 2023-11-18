from setuptools import setup

setup(
    name='dsctutorial',
    version='1.0.0',
    packages=['dsctutorial'],
    install_requires=[
        'genrisk @ git+https://github.com/airi-industrial-ai/genrisk',
        'gdown',
    ],
)

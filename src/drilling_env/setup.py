from setuptools import setup, find_packages

setup(
    name="drilling_env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gym>=0.21.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A drilling simulation environment following the OpenAI Gym interface",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="reinforcement-learning drilling-optimization gym-environment",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 
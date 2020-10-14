import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mltools-harwiltz',
    version='0.0.1',
    author='Harley Wiltzer',
    author_email='harley.wiltzer@mail.mcgill.ca',
    description='A package to simplify common ML operations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/harwiltz/mltools",
    packages=['mltools'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

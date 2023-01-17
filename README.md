# Measuring Quality of Life
This is a repository to track my research in measuring quality of life from satellite images using machine learning.

## Getting Started
#### Install Dependencies
This repository makes use of [Poetry](https://python-poetry.org/) which is a python package management solution. Follow the installation instructions [here](https://python-poetry.org/docs/#installation).

Once you have Poetry installed, you can run the following to install all required dependencies:
```shell
poetry install --no-root
```

#### Initialise Script
A script is included to do some initial setup such as preventing commits of outputs/metadata from jupyter notebooks.

In the root of the repo, execute:
```shell
chmod u+x init.sh
./init.sh
```
This currently:
- Strips jupyter notebook of output and metadata before committing
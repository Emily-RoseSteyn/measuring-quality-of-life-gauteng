# Measuring Quality of Life
This is a repository to track my research in measuring quality of life from satellite images using machine learning.

There are currently two primary objectives of this repository:
- Reproducing previous work by [Yeh et al](https://github.com/chrisyeh96/africa_poverty_clean) in a South African Context.
- Extending previous research by investigating measuring other socioeconomic indicators that describe "quality of life" more holistically.

For simplicity, these objectives are separated into 2 folders. The readmes for each should be a sufficient guide to what has been done.
- [01-reproduction](/01-reproduction/README.md)
- [02-extension](/02-extension/README.md)

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


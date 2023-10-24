# Beyond Poverty: Measuring Quality of Life from Satellite Images in Gauteng
Previous work has shown that asset wealth and consumption expenditure (recognised indicators of poverty) can be estimated from satellite images.

**Question**: Can other socioeconomic indicators be estimated from satellite images that represent a more holistic picture of quality of life?

- Focus on smaller area: Gauteng
- Start with satellite data that is lower res from Landsat
  - Then compare to higher res data from Planet
- First get asset wealth and income with nightlight proxy
- Consider other proxies

This is a repository to track my research in measuring quality of life from satellite images using machine learning.

The primary objective of this repository is to:
- Extend previous research by investigating measuring other socioeconomic indicators that describe "quality of life" more holistically in Gauteng.

## Getting Started
#### Install Dependencies
This repository makes use of [Poetry](https://python-poetry.org/) which is a python package management solution. Follow the installation instructions [here](https://python-poetry.org/docs/#installation).

Once you have Poetry installed, you can run the following in the root directory to install all required dependencies:
```shell
poetry install
```

#### Pre-commit Hooks
[Pre-commit](https://pre-commit.com/) is used to keep code clean and to make use of [DVC](https://github.com/iterative/dvc) on commits and pushes. You can take a look at what hooks run in the [config](./.pre-commit-config.yaml).

To ensure git hooks are setup for pre-commit, run:
```shell
pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit
```

#### DVC
[Data Version Control](https://github.com/iterative/dvc) is a cool tool that helps run end-to-end pipelines and track metrics and models at different checkpoints.

To make use of this functionality, you need to run the following in the root directory:
```shell
dvc install
```

This will ensure that:
* On checkout of a commit, any associated files are pulled from the relevant remote.
* On pushing, any files added to DVC will also be pushed.
* On committing, a check is run for the diff in local and remote DVC.

**DVC Remote Storage**

Contact [emilyrosesteyn@gmail.com](mailto:emilyrosesteyn@gmail.com) to get access to the existing remote storage container to access previous models and results.

Alternatively, if you are forking this repository, follow the instructions on the DVC docs to [add a remote](https://dvc.org/doc/command-reference/remote/add).


## Data Retrieval
- **Gauteng City Region Observatory Data**
  - Download GCRO dataset for 2020-2021 from [Data First](https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/874) portal.
  - Unzip folder
  - Rename the household DTA file to `gcro-2021.dta`
  - Copy DTA file to the [data/surveys](data/surveys) directory
  - Repeat this for 2014-2015 and 2017-2018

Your data directory should now look like:
```
data
  surveys
    gcro-2015.dta
    gcro-2018.dta
    gcro-2021.dta
```


[//]: # (TODO: Add years for dataset)
[//]: # (TODO: Add .env configuration)

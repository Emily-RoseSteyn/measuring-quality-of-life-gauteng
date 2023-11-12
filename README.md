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

Contact [emilyrosesteyn@gmail.com](mailto:emilyrosesteyn@gmail.com) to get access to the existing remote storage container hosted with Google Drive to access previous models and results. The credentials are in [.dvc/config](.dvc/config) but the app is unavailable unless you are an added test user.

Once added, when prompted,
Google may ask you to confirm giving access to the DVC remote storage app to manage all Gdrive files.
However, the scope of the app has been limited on GCP to only manage DVC's own created files.

[//]: # (See OAuth Scopes in docs - https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended and scopes on api consent window in GCP)

Alternatively, if you are forking this repository, follow the instructions on the DVC docs to [add a remote](https://dvc.org/doc/command-reference/remote/add).

## Pipeline
1. Data download
2. Data preprocessing
3. Training
4. Evaluation


[//]: # (TODO: Add years for dataset)
[//]: # (TODO: Add .env configuration)

## Gotchas
- [Reinstalling poetry environment](https://stackoverflow.com/questions/70064449/how-to-force-reinstall-poetry-environment)
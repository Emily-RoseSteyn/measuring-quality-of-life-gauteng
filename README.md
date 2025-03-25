# Beyond Poverty: Measuring Quality of Life from Satellite Images in Gauteng

Previous work has shown that asset wealth and consumption expenditure (recognised indicators of poverty) can be
estimated from satellite images.

**Question**: Can other socioeconomic indicators be estimated from satellite images that represent a more holistic
picture of quality of life?

- Focus on smaller area: Gauteng
- Start with satellite data that is lower res from Landsat
    - Then compare to higher res data from Planet
- First get asset wealth and income with nightlight proxy
- Consider other proxies

This is a repository to track my research in measuring quality of life from satellite images using machine learning.

The primary objective of this repository is to:

- Extend previous research by investigating measuring other socioeconomic indicators that describe "quality of life"
  more holistically in Gauteng.

## Getting Started

#### Python Version

This repo relies on Python 3.10 or greater. Take a look at [pyenv](https://github.com/pyenv/pyenv) for managing python
environments.

#### Install Dependencies

This repository makes use of [Poetry](https://python-poetry.org/) which is a python package management solution. Follow
the installation instructions [here](https://python-poetry.org/docs/#installation).

Once you have Poetry installed, configure it so that the virtual environment is installed inside your project directory:

```shell
poetry config virtualenvs.in-project true
```

You can then run the following in the root directory to install all required dependencies:

```shell
poetry install
```

Whenever you want to activate your shell to make use of the venv associated, you'll need to run poetry shell. To
install:

```shell
poetry self add poetry-plugin-shell
```

Then you can run:

```shell
poetry shell
```

#### Pre-commit Hooks

[Pre-commit](https://pre-commit.com/) is used to keep code clean and to make use
of [DVC](https://github.com/iterative/dvc) on commits and pushes. You can take a look at what hooks run in
the [config](./.pre-commit-config.yaml).

To ensure git hooks are setup for pre-commit, run:

```shell
pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit
```

#### DVC

[Data Version Control](https://github.com/iterative/dvc) is a cool tool that helps run end-to-end pipelines and track
metrics and models at different checkpoints.

To make use of this functionality, you need to run the following in the root directory:

```shell
dvc install
```

This will ensure that:

* On checkout of a commit, any associated files are pulled from the relevant remote.
* On pushing, any files added to DVC will also be pushed.
* On committing, a check is run for the diff in local and remote DVC.

**DVC Remote Storage**

Contact [emilyrosesteyn@gmail.com](mailto:emilyrosesteyn@gmail.com) to get access to the existing remote storage
container hosted with Google Drive to access previous models and results. The credentials are
in [.dvc/config](.dvc/config) but the app is unavailable unless you are an added test user.

Once added, when prompted,
Google may ask you to confirm giving access to the DVC remote storage app to manage all Gdrive files.
However, the scope of the app has been limited on GCP to only manage DVC's own created files.

In cases where you cannot interactively authorise with Google (e.g. a compute engine, cluster node, etc), you will need
a [credential file](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-service-accounts)
from GCP. Again, please contact [emilyrosesteyn@gmail.com](mailto:emilyrosesteyn@gmail.com).

[//]: # (See OAuth Scopes in docs - https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended and scopes on api consent window in GCP)

Alternatively, if you are forking this repository, follow the instructions on the DVC docs
to [add a remote](https://dvc.org/doc/command-reference/remote/add).

**DVC Studio**

In order to track experiments, DVC Studio can be used. To set this up, one can run `dvc studio login --no-open` and
follow
the instructions. The code in train and evaluation steps uses Live to track and save metrics and plots. This is synced
with DVC Studio if enabled.

## Pipeline

[//]: # (TODO: Document this more https://dvc.org/doc/user-guide/experiment-management)

1. [Data download](./data/README.md)
2. Data preprocessing
3. Training
4. Evaluation

[//]: # (TODO: Add years for dataset)

[//]: # (TODO: Add .env configuration)

## Gotchas

#### DVC

- **Credential Error on Push**
    - If you have been gone a while from the repo, your DVC Google credentials might expire and you will get the below
      error on a `git push`:
      ```text
        ERROR: unexpected error - failed to authenticate GDrive: Access token refresh failed: invalid_grant: Bad Request
      ```
    - To resolve, delete the [`gdrive credentials`](/.dvc/gdrive-credentials.json) file and retry `git push`. This will
      regenerate a new credentials file and the push should work.
- **DVC exp config error**
    - I had the following error when I first started using DVC experiments
      ```text
        ERROR: configuration error - config file error: extra keys not allowed @ data['exp']['auto_push']
      ```
    - I resolved it by switching off autopush `dvc config --global -u exp.auto_push`
    - However, this means that it is imperative to ensure experiments are pushed from remote locations

#### Poetry

[//]: # (TODO: Mpi4py)

- **Reinstall poetry packages**
    - [Reinstalling poetry environment](https://stackoverflow.com/questions/70064449/how-to-force-reinstall-poetry-environment)
- **No sudo but need different version of python?**
    - If you don't have sudo access on a linux system, poetry and python can be a bit weird. You can
      use [pyenv](https://github.com/pyenv/pyenv) for managing python environments without sudo access.
    - For example, on a system that has `python3.8`, one can run `pyenv install 3.11` to install a newer version of
      python without requiring sudo access.
    - Keep in mind that poetry does not install python but rather references available versions of python and creates
      the right virtual environment with the right executable before installing packages.
- **Poetry install hangs**
    - I ran into an issue where poetry hangs when installing on cluster without sudo access similar
      to [this one](https://github.com/python-poetry/poetry/issues/8623). The solution
      of `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` worked for me.
- **Repeated warning: _The currently activated Python version 3.8.10 is not supported by the project_**
    - If this warning keeps showing up, you might have a system python/poetry mismatch.
    - I seemed to resolve this on a linux system, by ensuring the right python version is installed with pyenv and then
      setting poetry to use that version. E.g. for python 3.11:
        - `pyenv install 3.11`
        - In the project directory: `pyenv local 3.11`
        - `poetry env use 3.11`
- **_ImportError: No module named '\_sqlite3_**
    - This is because of a missing underling sqlite package on a linux system. Python binds to the system sqlite package
      through `_sqlite3`.
    - Get a `sudo` user to run: `sudo apt install libsqlite3-dev`
    - NB! Reinstall python with pyenv: `pyenv install <VERSION>`.
        - Say yes if prompted to install over an existing version.
    - [Ref](https://github.com/pyenv/pyenv/issues/678#issuecomment-312159387)

#### Jupyter

- **Running on cluster**
    - If you have the project setup on a remote cluster, you can run a jupyter notebook from a node by running:
        - `jupyter notebook --no-browser --port 8888 --ip 0.0.0.0`
    - To connect to this notebook instance from your local machine, you need to port forward the node port 8888 to your
      machine:
        - `ssh -fNL 8888:<NODE_NAME>:8888 <user>@<cluster-ip>`
        - Where `NODE_NAME` is the node name as seen from the head node that you connect to at `user@cluster-ip`

#### Tensorflow

- **Tensorboard**
    - To view logs from tensorflow and access tensorboard functionality, you can run:
      `tensorboard --logdir logs/scalars --host=0.0.0.0`
    - This must be run on the node where the logs are generated as logs are not committed
- **Multiple processes with GPU**
    - If you are running multiple processes with a GPU in parallel, you want to ensure that each process has sufficient
      memory allocated. By default, tensorflow seems to allocate all the GPU memory to a running process and so
      subsequent processes that are started in parallel will have insufficient memory.
    - To solve this, you can make use of the memory growth flag for tensorflow which "attempts to allocate only as much
      GPU memory as needed for the runtime allocations". Minimum memory is initially allocated and as more GPU memory is
      needed, the GPU memory is extended for the specific tensorflow process. See
      more [here](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth).
    - Code:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
```

[//]: # (TODO: Rename outputs -> output)
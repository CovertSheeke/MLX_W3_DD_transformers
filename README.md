<!-- $ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm ~/miniconda3/miniconda.sh
$ source ~/miniconda3/bin/activate
$ conda init --all
$ conda create --name trf python=3.11 -y
$ conda activate trf
$ pip install datasets torch more-itertools wandb -->

Use the above to setup conda.

install uv using the command

apt install uv  (brew install uv [for mac])

then use 

uv sync

to bring your current uv venv into alignment with the uv.lock and pyproject.toml files

you must then run source 

source .venv/bin/activate

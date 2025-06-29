# Set up
Thre options, uv, conda or mamba.

## UV
install uv using the command

apt install uv  (brew install uv [for mac])

then use 

uv syn

to bring your current uv venv into alignment with the uv.lock and pyproject.toml files

you must then run source 

source .venv/bin/activate

## Conda
<!-- $ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm ~/miniconda3/miniconda.sh
$ source ~/miniconda3/bin/activate
$ conda init --all
$ conda create --name trf python=3.11 -y
$ conda activate trf
$ pip install datasets torch more-itertools wandb -->

## Mamba
Install mamba, then run:
```
mamba env create -f env.yml

eval "$(mamba shell hook --shell bash)"

mamba activate pytorch_env
```
The UV env can then also be set up from within the mamba env, but isn't needed

## Test if the GPU can be found by pytorch:
Run the test script:

```python misc/gpu_test.py```

It should print:
```
CUDA available?: True
Number of GPUs: 1
```
If not, it could be a number of reasons, often it is because the right cuda and cudnn version are not installed for the model of GPU.


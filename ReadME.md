To run, you need to install architectural style classification dataset from 
https://sites.google.com/site/zhexuutssjtu/projects/arch (Removed),
-> A similar one can be found on https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset.

Prior to installation you may create a new virtual environment with one of below.
```
    python3.8 -m venv my_env_name
    source my_env_name/bin/activate
```
If you specify a python version with conda it will come with its own pip installed.
```
    conda create --name my_env_name  python=3.8
    conda activate my_env_name
```

To run, install requriements by;
```
    pip install -r reqirements.txt 
    pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102  -> With Cuda
    pip install torch torchvision -> Without Cuda
```

Then just set the options and run the main:
```
    python main.py
```


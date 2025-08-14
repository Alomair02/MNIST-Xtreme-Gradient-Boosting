# Instructions

## 1. Creating a Conda Environment
Before proceeding, create and activate a conda environment with Python 3.12.10:
```bash
conda create --name digit-recognizer-env python=3.12.10 -y
conda activate digit-recognizer-env
```

## 2. Unzipping the Dataset
Unzip the `digit-recognizer.zip` file into the `data` directory by running the following command:
```bash
unzip digit-recognizer.zip -d data
```

## 3. Installing Requirements
Install the required dependencies using the `requirements.txt` file. You can use either `conda` or `pip` as the package installer:
```bash
pip install -r requirements.txt
# or
conda install --file requirements.txt
```

## 4. Running the Script
Run the `main.py` script with the following example arguments:
```bash
python main.py --n_splits 5 --num_rounds 500 --data_directory data
```
# Audio Number Classification
## About project

This project was created as part of homework on MLOps.
The goal of the project is to study and use tools such as poetry, dvc, pre-commit, mlflow, wandb.

The task of machine learning is to classify numbers from 0 to 9 spoken in audio recordings.

## About dataset

The dataset was taken from a competition as part of homework for the Additional Chapters of ML/DL.
The dataset is available at the [link](https://www.kaggle.com/competitions/mlds-asr-competition-on-toloka-labelling/data).

It consists of training, validation and test data.

Training and validation data contains short audio recordings in `.mp3` format and `.csv` files with annotations.
The test data contains only audio recordings.

## Project structure

`audio_number_classification` package contains general services for train and test model:
1. `dataset.py` contains a AudioDataset, which adapt dataset for working with torchaudio
2. `augmentation.py`contains AudioTransformer for mixing and augmentate audio
3. `model.py` - contains CNN based pytorch (inherit nn.Module) for classifying number in audio
4. `predictor.py` - contains Predictor with static method which predict number in audio
5. `trainer.py` - contains Trainer. Trainer use hyperparameters, logger and other
6. `logging` package contains abstract Logger for different implementations and use in DI then. Also contains Decorator pattern implementation.
7. `factoring` package contains Factory Method pattern implementation for general services

`train.py` and `infer.py` are client part which use services in `audio_number_classification` package.

## Run project

1. Clone repository

```bash
git clone https://github.com/olga-golovkina/audio-number-classification.git
cd audio-number-classification
```
2. Create a new virtualenv
```bash
python -m venv venv
```

3. Install using poetry packages
```bash
poetry install
```

4. Install pre-commit
```bash
pre-commit install
```

5. Check repos by pre-commit rules
```bash
pre-commit run -a
```

6. Run training
```bash 
python train.py
```
or 
```bash 
venv/scripts/python.exe train.py
```

7. Run inference
```bash 
python infer.py
```
or 
```bash 
venv/scripts/python.exe infer.py
```



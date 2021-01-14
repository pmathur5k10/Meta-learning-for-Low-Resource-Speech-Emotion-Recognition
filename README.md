# Meta-learning-for-Low-Resource-Speech-Emotion-Recognition
Meta-learning for Low-Resource Speech Emotion Recognition

Code for MAML, Multi-task learning and Transfer Learning is present in the respective folders.

## Requirements
To install requirements, run the given commands:
`conda env create -f environment.yml`
`conda activate SER`

## Transfer Learning 
- `cd Transfer-Learning`
- Modify datasets list in utils
- Run training: `python train.py`

## Multi-Task Classification 
- `cd multi-task-classification`
- Modify train and test datasets list in utils
- Run: `python main.py --<train/eval> --save_path <save_path>`

## Meta Learning 
- `cd MAML-in-pytorch`
- Modify train and test datasets list in utils
- Run: `python train.py --model_dir <save_path>`

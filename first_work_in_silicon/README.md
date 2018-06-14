# Text Classification with Tensorflow

## Requirements
```
python3
tensorflow >= 1.7
```
## Task
Given a sentence, assign a label according to its' content.
```
办理前需要先了解下您的个人信息，请您如实告知。请问您现在居住地是在甘肃省内吧？__user__嗯是的 --- 1
```

## data & preprocess
The data provided in the `data/` directory is a csv file

In `data_util.py` I provide some funtions to process the csv file.

## Usage
This contains several steps:
1. Before you can get started on training the model, you mast run
```
python data_util.py
```

2. After the dirty preprpcessing jobs, you can try running an training experiment with some configurations by:
```
python little_try.py train
```

3. You can also run an evaluation by:
```
python little_try.py evaluate
```
After the program is done, the you can run:
```
python my_evaluate.py
```
to get the result in test set.


## Folder Structure
```
├── data            - this fold contains all the data
│   ├── train
│   ├── dev
│   ├── test
│   ├── vocab
|   ├── vec
├── model           - this fold contains the pkl file to restore
├── little_try.py   - main entrance of the project
├── data_util.py    - preprocess the data
├── batch_data.py   - data generator
├── my_evaluate.py  - evaluate the performance of the model in test set   
```

## To do
1. Still need parameters searching.
2. Need structure changing to satisfy parameters chosing.
3. Make codes nicer.

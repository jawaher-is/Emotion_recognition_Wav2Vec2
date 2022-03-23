"""## Prepare Data

For this particular example, we use [Acted Emotional Speech Dynamic Database – AESDD](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/) provided by Multidisciplinary Media & Mediated Communication Research Group ([M3C](http://m3c.web.auth.gr/)).

The Acted Emotional Speech Dynamic Database (AESDD) is a publically available speech emotion recognition dataset that contains utterances of acted emotional speech in the Greek language for five different emotions `sadness`, `disgust`, `happiness`, `anger`, and `fear`.

The dataset consists of directories of emotions; each folder includes specific emotions. We need to loop over directories and save the paths related to each class based on the directory name.

```bash
.
├── Tools\ and\ Documentation
│   ├── ESTrainer.mlapp
│   ├── Speech\ Emotion\ Recognition\ Adapted\ to\ Multimodal\ Semantic\ Repositories_documentation.pdf
│   ├── Speech\ Emotion\ Recognition\ for\ Performance\ Interaction.pdf
│   └── readme.txt
├── anger
│   ├── a01\ (1).wav
│   ├── a01\ (2).wav
│   ├── ...
├── disgust
│   ├── d01\ (1).wav
│   ├── d01\ (2).wav
│   ├── ...
├── fear
│   ├── f01\ (1).wav
│   ├── f01\ (2).wav
│   ├── ...
├── happiness
│   ├── h01\ (1).wav
│   ├── h01\ (2).wav
│   ├── ...
└── sadness
    ├── s01\ (1).wav
    ├── s01\ (2).wav
    ├── ...

6 directories, 609 files
```

Since the dataset is hosted on [Mega](https://mega.nz/#F!0ShVXY7C!-73kVoK05OjTPEA95UUvMw) and unable to access it directly, we upload the dataset on the G-drive to make this tutorial accessible.
"""

# Download the dataset from

# !mkdir -p ./content/data
# !gdown https://drive.google.com/uc?id=1_IAWexEWpH-ly_JaA5EGfZDp-_3flkN1
# !unzip -q aesdd.zip -d /content/data/
# !mv "/content/data/Acted Emotional Speech Dynamic Database/" /content/data/aesdd/

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys

data = []

print("DATA PREPERATION")
# Paths for data
Ravdess = "./content/data/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "./content/data/kaggle/input/cremad/AudioWAV/"
Tess = "./content/data/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
Savee = "./content/data/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"

exceptions = 0
#1. RAVDESS DataFrame
ravdess_data = []
for path in tqdm(Path(Ravdess).glob("**/*.wav")):
    name = str(path).split('/')[-1].split('.')[0]
    label = name.split('-')[2]

    try:
        # avoid broken files
        s = torchaudio.load(path)
        ravdess_data.append({
            "name": name,
            "path": path,
            "emotion": label
        })
    except Exception as e:
        # print(str(path), e)
        exceptions+=1
        pass

ravdess_df = pd.DataFrame(ravdess_data)
ravdess_df.emotion.replace({'01':'neutral', '02':'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}, inplace=True)
# ravdess_df.head() # do not display on hpc

# 2. CREMA DATAFRAME
crema_data = []
for path in tqdm(Path(Crema).glob("*.wav")):
    name = str(path).split('/')[-1].split('.')[0]
    label = name.split('_')[2]

    if label == 'SAD':
        label = 'sad'
    elif label == 'ANG':
        label = 'angry'
    elif label == 'DIS':
        label = 'disgust'
    elif label == 'FEA':
        label = 'fear'
    elif label == 'HAP':
        label = 'happy'
    elif label == 'NEU':
        label = 'neutral'
    else:
        label = 'unknown'
        print('Unknown label detected in ', path)

    try:
        # avoid broken files
        s = torchaudio.load(path)
        crema_data.append({
            "name": name,
            "path": path,
            "emotion": label
        })
    except Exception as e:
        # print(str(path), e)
        exceptions+=1
        pass

crema_df = pd.DataFrame(crema_data)
# crema_df.head()

# 3. TESS DATASET
tess_data = []
for path in tqdm(Path(Tess).glob("**/*.wav")):
    name = str(path).split('/')[-1].split('.')[0]
    label = name.split('_')[-1]
    if label =='ps': label = 'surprise'

    try:
        # avoid broken files
        s = torchaudio.load(path)
        tess_data.append({
            "name": name,
            "path": path,
            "emotion": label
        })
    except Exception as e:
        # print(str(path), e)
        exceptions+=1
        pass

tess_df = pd.DataFrame(tess_data)
# tess_df.head()
# tess_df["emotion"].unique()

# 4. Savee DATASET
savee_data = []
for path in tqdm(Path(Savee).glob("*.wav")):
    name = str(path).split('/')[-1].split('.')[0]
    label = name.split('_')[1][:-2]

    if label == 'a':
        label = 'angry'
    elif label == 'd':
        label = 'disgust'
    elif label == 'f':
        label = 'fear'
    elif label == 'h':
        label = 'happy'
    elif label == 'n':
        label = 'neutral'
    elif label == 'sa':
        label = 'sad'
    elif label == 'su':
        label = 'surprise'
    else:
        label = 'unknown'
        print('Unknown label detected in ', path)

    try:
        # avoid broken files
        s = torchaudio.load(path)
        savee_data.append({
            "name": name,
            "path": path,
            "emotion": label
        })
    except Exception as e:
        # print(str(path), e)
        exceptions+=1
        pass

savee_df = pd.DataFrame(savee_data)
# savee_df.head()

df = pd.concat([ravdess_df, crema_df, tess_df, savee_df], axis = 0)

print(f"Step 0: {len(df)}") # 12162

df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
df = df.dropna(subset=["path"])
df = df.drop("status", 1)
print(f"Step 1: {len(df)}") # 12162

df = df.sample(frac=1)
df = df.reset_index(drop=True)
df.head()

"""Let's explore how many labels (emotions) are in the dataset with what distribution."""
print("Labels: ", df["emotion"].unique())
print()
df.groupby("emotion").count()[["path"]]

"""For training purposes, we need to split data into train test sets; in this specific example, we break with a `20%` rate for the test set."""

save_path = "./content/data"

train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["emotion"]) # stratify for a balanced number of examples for each class

train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["emotion"])

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

print(train_df.shape)
print(test_df.shape)

# TODO: think of augmenting the data. Adding noise/ strech

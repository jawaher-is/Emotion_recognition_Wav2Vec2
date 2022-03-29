# Emotion_recognition_Wav2Vec2

Speech Emotion Recognition on [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), [CREMA](https://www.kaggle.com/datasets/ejlok1/cremad), [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess), and [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) datasets.

Reference:
- https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb


#### Evaluation results
|         | precision | recall | f1-score | support |
| ------- | ---- | ---- | ---- | ----|
| angry   | 0.84 | 0.93 | 0.88 | 385
| calm    | 0.56 | 0.95 | 0.71 | 38
| disgust | 0.78 | 0.78 | 0.78 | 385
| fear    | 0.86 | 0.68 | 0.76 | 385
| happy   | 0.83 | 0.75 | 0.79 | 385
| neutral | 0.80 | 0.89 | 0.85 | 340
| sad     | 0.72 | 0.75 | 0.73 | 385
| surprise| 0.91 | 0.93 | 0.92 | 130
|
| accuracy     |      |      | 0.80 | 2433
| macro avg    | 0.79 | 0.83 | 0.80 | 2433
| weighted avg | 0.81 | 0.80 | 0.80 | 2433

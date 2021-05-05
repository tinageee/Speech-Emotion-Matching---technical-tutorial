# How to Compare the Emotions from Acoustic and Linguistic Features of a Speech

Although the literature on sentiment and emotion analyses is extensive, few rigorous investigations have documented the emotion synchrony between different modalities.

From this article, you will understand how to use this python package to extract the sentiments from the speech and compare the emotion expressed from acoustic and conveyed from linguistic.

## Where can we use it?

### Deception detection:
Although deceivers can apply high-level control for behavioral management, many non-verbal behaviors are driven unconsciously. They cannot be perfectly controlled(Buller and Burgoon 1996), so deceivers can still leak deceptive cues through their voice and facial expression. Based on the premise that lying is cognitively more demanding than truth-telling, existing deception detection studies have examined behavioral patterns between deceivers and truth-tellers(Ekman,1969). Driving from the same theory, this excessive cognitive load could promote inconsistency between the emotion expressed by the deceiver’s linguistic and acoustic features.

<br/>**Reference**:
<br/>Buller, D.B. and Burgoon, J.K. 1996. “Interpersonal deception theory,” Communication theory (6:3), pp.203-242.
<br/>Ekman, P., & Friesen, W. V. 1969. Nonverbal leakage and clues to deception. Psychiatry, 32(1), 88–106

## Model basic information
### Usage
_Test whether the emotion in the speaker's language and audio aligns_
### Natural languages supported
_English_

## Steps:
1. Identify the emotions from the text
2. Identify the emotions from the audio
3. Compare two emotion tags

### Step 0: Set a Proper Environment

### Python Packages
- **librosa==0.6.3**
- **numpy**
- **pandas**
- **soundfile==0.9.0**
- **wave**
- **sklearn**
- **tqdm==4.28.1**
- **matplotlib==2.2.3**
- **pyaudio==0.2.11**
Install these libraries by the following command:
```
pip install -r requirements.txt
```

### Step 1: Identify the Emotions from the Text
Python Package:[Text2Emotion](https://pypi.org/project/text2emotion/)
- Processes any textual data, recognizes the emotion embedded in it and provides the output in the form of a dictionary.
- Well suited with 5 basic emotion categories such as **Happy, Angry, Sad, Surprise, and Fear**.
- The output will be in the form of a dictionary where keys as emotion categories and values as emotion scores.
```python
import text2emotion as  te
text="avsydgaiusdgajhdlakdj;ksj"
text_emotion=te.get_emotion(text)
text_emotion
```
** output:**
```
{'Happy': 0.0, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.8, 'Fear': 0.2}
```

### Step 2: Identify the Emotions from the Audio

Codes and data in this section are adapted and modified from [x4nth055](https://github.com/x4nth055).

<br/>_Project:[Emotion Recognition Using Speech](https://github.com/x4nth055/emotion-recognition-using-speech)_
<br/>_Copyright (c) 2019 x4nth055_
<br/>_License (MIT) https://github.com/x4nth055/emotion-recognition-using-speech/blob/master/LICENSE_

Model development Dataset:
This  used 2 datasets (including this repo's custom dataset) which are downloaded and formatted already in `data` folder:
- [**RAVDESS**](https://zenodo.org/record/1188976) : The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.
- [**TESS**](https://tspace.library.utoronto.ca/handle/1807/24487) : **T**oronto **E**motional **S**peech **S**et that was modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966). A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years).

Data Preprocessing and Feature Extraction:
- There are 9 emotions available in the dataset: "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise) and "boredom". Here we only use 5 basic emotion categories: **Happy, Angry, Sad, Surprise, and Fear**.
- Feature extraction is the main part of the speech emotion recognition system. It is basically accomplished by changing the speech waveform to a form of parametric representation at a relatively lesser data rate.

```python
from deep_emotion_recognition import DeepEmotionRecognizer
# initialize instance
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=['happy','angry', 'ps', 'sad', 'fear'],
                                n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
# train the model
deeprec.train()
# get the accuracy
print(deeprec.test_score())
# predict angry audio sample
prediction = deeprec.predict('data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav')
print(f"Prediction: {prediction}")
```







I hope you got the idea about the basic functionalities provided by this program. If you have any questions, you are welcome to reach me by [ge1@email.arizona.edu](ge1@email.arizona.edu)
All code shown here is in [this GitHub repository](https://github.com/tinageee/technical-tutorial.git). Feel free to leave a star!

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```


# How to Compare the Emotions from Speech's Acoustic and Linguistic Data

Although the literature on sentiment and emotion analyses is extensive, few rigorous investigations have documented the emotion synchrony between different modalities.

From this article, you will understand how to use this python package in extracting the emotions from the speech and comparing the emotion expressed from acoustic and expressed from linguistic.

## Where can we use it?

### Deception detection:
Although deceivers can apply high-level control for behavioral management, many non-verbal behaviors are driven unconsciously and cannot be perfectly controlled(Buller and Burgoon 1996), so deceivers can still leak deceptive cues through their voice and facial expression. Based on the premise that lying is cognitively more demanding than truth-telling, existing deception detection studies have examined behavioral patterns between deceivers and truth-tellers(Ekman,1969). Driving from the same theory, this excessive cognitive load could promote inconsistency between the emotion expressed by the deceiver’s linguistic and acoustic features.

<br/>**Reference**:
<br/>Buller, D.B. and Burgoon, J.K. 1996. “Interpersonal deception theory,” Communication theory (6:3), pp.203-242.
<br/>Ekman, P., & Friesen, W. V. 1969. Nonverbal leakage and clues to deception. Psychiatry, 32(1), 88–106

## Model basic information
### Usage
_Test whether the emotion in speaker's language and audio aligns_
### Natural languages supported
_English_

## Steps:
1. Identify the emotion from the text
2. Identify the emotion from the audio
3. Compare two emotion tags

### Step 1: Identify the emotion from the text
Python  Package:[Text2Emotion](https://pypi.org/project/text2emotion/)
- Processes any textual data, recognizes the emotion embedded in it, and provides the output in the form of a dictionary.
- Well suited with 5 basic emotion categories such as **Happy, Angry, Sad, Surprise, and Fear**.
- The output will be in the form of dictionary where keys as emotion categories and values as emotion score.
```python
import text2emotion as  te
text="avsydgaiusdgajhdlakdj;ksj"
text_emotion=te.get_emotion(text)
text_emotion
```
**Output:**
```
{'Happy': 0.0, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.8, 'Fear': 0.2}
```







I hope you got the idea about the basic functionalities provided by this progarm. If you have any questions, you are welcome to reach me by [ge1@email.arizona.edu](ge1@email.arizona.edu)
All code shown here is in [this github repository](https://github.com/tinageee/technical-tutorial.git). Feel free to leave a star!

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


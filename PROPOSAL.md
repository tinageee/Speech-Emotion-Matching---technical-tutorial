### Tool name

_text and audio based emotion congruency checker_

### URL (tutorial)

_https://tinageee.github.io/technical-tutorial/_

### URL (tool)

Vader:https://pypi.org/project/vaderSentiment/<br/>
OpenSMILE: https://www.audeering.com/opensmile/<br/>
<br/>
I am still exploring other sentiment analysis tools. Vader only provides emotion labels with positive, negative, and neutral. I would like to try with more tags, ie. sad, happy, anger, etc._

### License

_License name and link if available_

### Usage

_Test whether the emotion in speaker's language and audio aligns_

### Natural languages supported

_English_

### Proposal

Domain: affective computing<br/>
Potential Application: deception detection<br/>
<br/>
Based on the premise that lying is cognitively more demanding than truth-telling, existing deception detection studies have examined behavioral patterns between deceivers and truth-tellers(Ekman,1969). Driving from the same theory, this excessive cognitive load could promote inconsistency between the emotion expressed by the deceiver’s linguistic, acoustic, and facial features. <br/>
For this NLP class' proposal, I would like to examine only two fore-mentioned features on pre-processed data. Therefore, in this technical tutorial, I will present how to examine whether the emotion in linguistic and audio aligns in an audio clip. <br/>
<br/>
Ekman, P., & Friesen, W. V. 1969. Nonverbal leakage and clues to deception. Psychiatry, 32(1), 88–106 <br/>
 
#### Implement plan:
1. Identify the emotion from the text
2. Identify the emotion from the audio
3. Compare two emotion tags
 
#### What makes this tool unique/different from similar offerings?
 
The studies on emotional recognization tools are extensive. However, to the best of my knowledge, no tool compares the emotion from different modalities.
 
#### Can I evaluate how well the tool performs on some data?
 
1)The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)<br/>
2) generate test data with ground truth. 


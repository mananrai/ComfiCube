from keras.models import load_model
import numpy as np
from emo_utils import *

import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

## obtain path to "english.wav" in the same folder as this script
#from os import path
#AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")
## AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
## AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")
#
## use the audio file as the audio source
#r = sr.Recognizer()
#with sr.AudioFile(AUDIO_FILE) as source:
#    audio = r.record(source)  # read the entire audio file

# recognize speech using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    words = r.recognize_google(audio)
    print("Google Speech Recognition thinks you said " + words)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    return X_indices


model = load_model('checkpoints/my_model.h5')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
maxLen = 10

#x_test = np.array(['not feeling happy','I love you so much','let us play ball'])
x_test = np.array([words])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(x_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    print(' prediction: ' + x_test[i] + label_to_emoji(num).strip())


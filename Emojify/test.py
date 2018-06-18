from keras.models import load_model
import numpy as np
from emo_utils import *

import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

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
            if w not in word_to_index.keys():
                X_indices[i, j] = len(word_to_index)
            else:
                X_indices[i, j] = word_to_index[w]
            j = j + 1
    return X_indices

model = load_model('checkpoints/short_sentences_weights.h5')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.twitter.27B.50d.txt')
maxLen = 10 # Set to 34 for PopBots

# Use this for PopBots data
#x_test = np.array(['Hi Not I\'m Sir Laughs Bot',
#                   'I\'m here to help you deal with your stress',
#                   'Could you share something that\'s on your mind that is stressing you out',
#                   'Loads of work',
#                   'Ok tell me more about this situation',
#                   'Lots of school work',
#                   'Thank you for sharing',
#                   'That does sound stressful',
#                   'Ok let\'s try looking at this situation in a different light',
#                   'I want you to take a few minutes to come up with a joke about this situation',
#                   'Would you like an example',
#                   'No I\'m good',
#                   'Go for it',
#                   'Ha ha so much work',
#                   'Good joke',
#                   'Sometimes there are good things that happen even if the situation isn\'t the best',
#                   'Oh',
#                   'Did that help you to find something good (or at least funny) about the situation',
#                   'Perhaps',
#                   'I\'m glad Would you consider trying this strategy such as finding a joke in the future',
#                   'Maybe',
#                   'Do you think Sir Laughs Bot helps you to reduce stress Almost no help neutral or very helpful',
#                   'neutral',
#                   'Thank you for sharing with me I hope I\'ve been able to help',
#                   'Have a nice day',
#                   'Thank you'])

x_test = np.array(['stop messing around',
                   'any suggestions for dinner',
                   'I love taking breaks',
                   'you brighten my day',
                   'I boiled rice',
                   'she is a bully',
                   'Why are you feeling bad',
                   'I am upset',
                   'I worked during my birthday',
                   'My grandmother is the love of my life',
                   'enjoy your break',
                   'valentine day is near',
                   'I miss you so much',
                   'throw the ball',
                   'My life is so boring',
                   'she said yes',
                   'will you be my valentine',
                   'he can pitch really well',
                   'dance with me',
                   'I am starving',
                   'See you at the restaurant',
                   'I like to laugh',
                   'I will go dance',
                   'I like your jacket',
                   'i miss her',
                   'what is your favorite baseball game',
                   'Good job',
                   'I love to the stars and back',
                   'What you did was awesome',
                   'ha ha ha lol',
                   'I want to joke',
                   'go away',
                   'yesterday we lost again',
                   'family is all I have',
                   'you are failing this exercise',
                   'Good joke',
                   'You totally deserve this prize',
                   'I did not have breakfast'])


#x_test = np.array([words])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(x_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    print(' prediction: ' + x_test[i] + label_to_emoji(num).strip())


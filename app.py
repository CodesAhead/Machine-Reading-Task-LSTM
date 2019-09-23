from flask import Flask, render_template, request, session
import pickle
import re
import warnings
import numpy as np
from functools import reduce
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
# import pandas as pd
from keras import backend as K
from flask import render_template_string
import random
from functions import get_stories, tokenize, vectorize_context, word_index, indx_word
import tensorflow as tf

warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder="templates")


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/story')
def story_page():
    ftest = open('/home/anthony/Documents/project _(mom)/Test-pjt2.txt', 'rb')
    test_stories = get_stories(ftest)

    story = test_stories[random.randint(0, 1000)][0]
    # session['by_var'] = story
    randstory = ' '.join(story)

    return render_template('story.html', rand=randstory)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    # session['by_var'] = story
    model1 = pickle.load(open("/home/anthony/PycharmProjects/babi/model_babi.dat", "rb"))
    ftest = open('/home/anthony/Documents/project _(mom)/Test-pjt2.txt', "rb")
    test_stories = get_stories(ftest)
    story = test_stories[random.randint(0, 1000)][0]
    randstory = ' '.join(story)

    if request.method == 'POST':


        text = request.form['quer']
        query = 'Where is ' + text + ' ?'

        story = session.get('by_var', None)
        current_inp = (story, tokenize(query))
        wd = word_index()
        idx_word = indx_word()
        current_story, current_query = vectorize_context([current_inp], wd, 68, 4)
        current_prediction = model1.predict([current_story, current_query])
        current_prediction = idx_word[np.argmax(current_prediction)]
        p = current_prediction

    K.clear_session()
    return render_template('prediction.html', q=query ,predict=p)


if __name__ == '__main__':
    app.run(debug=True)

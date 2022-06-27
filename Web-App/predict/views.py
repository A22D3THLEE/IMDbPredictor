import string
import pandas as pd
import nltk
import spacy
nltk.download('punkt')
from nltk.corpus import stopwords
from django.shortcuts import render
from django.http import JsonResponse
from keras.utils.data_utils import pad_sequences
from predict.models import PredResults
from keras.models import load_model
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from predict.load_data import load

dictionary= load()


def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):
    if request.POST.get('action') == 'post':
      text = request.POST.get('review')
    if len(text) != 0:
      text = text.lower()

      for p in string.punctuation:
        if p in text:
            text = text.replace(p, '')

      nlp = spacy.load("en_core_web_sm")
      doc = nlp(text)
      lemma_tokens = []
      for token in doc:
        lemma_tokens.append(token.lemma_)
      lemmatize_text = ' '.join(map(str, lemma_tokens))

      tokenize_text = word_tokenize(lemmatize_text)

      stop_words = set(stopwords.words('english'))
      stop_words.remove('not')
      filtered_sentence = []
      for w in tokenize_text:
        if w not in stop_words:
            filtered_sentence.append(w)

      lenth = len(filtered_sentence)
      words_token = []
      for i in range(0, lenth):
        if filtered_sentence[i] in dictionary:
            words_token.append(dictionary[filtered_sentence[i]])
        else:
            continue

      lenth = len(words_token)
      final = []
      for i in range(0, lenth):
        if words_token[i] < 10000:
            final.append(words_token[i])

      if len(final) != 0:
        data = pad_sequences([final], maxlen=200)
        model = load_model('bestmodel.h5')
        result = model.predict(data)
        if result >= 0.7:
            classification = 'positive'
        elif 0.5 <= result < 0.7:
            classification = 'neutral'
        elif result < 0.5:
            classification = 'negative'
        rating = result*10
        PredResults.objects.create(text=request.POST.get('review'), classification=classification, rating=rating)
        return JsonResponse({'classification': classification}, safe=False)

      else:
        classification = 'incorrect'
        return JsonResponse({'classification': classification}, safe=False)
    else:
        classification = 'incorrect'
        return JsonResponse({'classification': classification}, safe=False)
def view_result (request):
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)
def view_info(request):
    return render(request, 'about.html')










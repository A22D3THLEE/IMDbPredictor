import string
import nltk
import pandas as pd
import spacy
from keras.saving.save import load_model
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences

# Введите текст
with open('text_file.txt') as file:
    text = file.read()
print(text)
# Получаем данные из БД
train = pd.read_csv('train.csv', header=None, names=['Review'])
reviews = train['Review']
# Создаем токенайзер
tokenizer = Tokenizer(num_words=10000)

# Составляем словарь из токенов (каждый токен равен частоте использования слова в отзывах)
tokenizer.fit_on_texts(reviews)
dictionary = tokenizer.word_index


# Приводим входящий отзыв к нижнему регистру
text = text.lower()

# Убираем пунктуацию
for p in string.punctuation:
    if p in text:
        text = text.replace(p, '')

# Лемматизируем (приводим каждое слово в его изначальную форму)

nlp = spacy.load("en_core_web_sm")  # Модель для лемматизации слов
doc = nlp(text)
lemma_tokens = []
for token in doc:
    lemma_tokens.append(token.lemma_)
lemmatize_text = ' '.join(map(str, lemma_tokens))

# Токенизируем отзыв (разбиваем каждое слово на токены)
tokenize_text = word_tokenize(lemmatize_text)

# Убираем стоп-слова
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
filtered_sentence = []
for w in tokenize_text:
         if w not in stop_words:
             filtered_sentence.append(w)

# Ищем слова, которые содержатся в нашем словаре, чтобы присвоить им подходящий индекс/токен по их частоте появления в отзывах
lenth = len(filtered_sentence)
words_token = []
for i in range(0, lenth):
    if filtered_sentence[i] in dictionary:
        words_token.append(dictionary[filtered_sentence[i]])
    else:
        continue

# Исключаем те токены, которые не были изучены нашей моделью (Модель изучила все токены, значение которых меньше 10к)
lenth = len(words_token)
final = []
for i in range(0, lenth):
    if words_token[i] < 10000:
        final.append(words_token[i])
# Составляем вектор наших токенов-слов. Это финальное преобразование перед передачей данных модели
if len(final) != 0:
       data = pad_sequences([final], maxlen=200)
else:
       data = "Data wrong!"


model = load_model('bestmodel.h5')
result = model.predict(data)
print(result[[0]])

print(text, "\n") # Исходник
print(lemmatize_text, "\n") # Лемматизация
print(tokenize_text, "\n") # Токенизация
print(filtered_sentence, "\n") # Стоп-слова4
print(words_token, "\n") # Итог
print(data, "\n") # Вектор для модели
print(lenth)

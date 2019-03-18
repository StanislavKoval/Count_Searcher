import pandas as pd
import numpy as np
import re
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

class preprocessor:

    #SETS
    abbreviation = ["рад", "мрад", "мкрад",  # угол
                    "см", "м", "мм", "мкм", "нм", "дм",  # метр
                    "кг", "мг", "мкг", "г", "т",  # вес
                    "мин", "ч", "сут", "с", "мс", "мкс", "нс",  # время
                    "л",  # объем
                    "гц", "ггц", "мгц", "кгц",  # Гц
                    "шт",  # кол-во
                    "ом", "а", "в", # эл-тех
                    "млн","тыс","млрд","руб", "ме",#денеж.ср.
                    "бит", "байт", "кбай", "мбайт", "гбайт","тбайт","мбайт"]#информ.

    stop_words_rus = set(stopwords.words('russian'))
    stop_words_eng = set(stopwords.words('english'))

    def prep_df(data_file):

        # token (only: word, word-word or bigrams)
        tokenizer = RegexpTokenizer('\w+-\w+|"\w+\s\w+"|\w+|"\w+\s\w+\s\w+"')  # for word and word-word and "word word" (as bigram)
        words = list()

        # making string type
        for column in data_file.columns:
            data_file[column] = data_file[column].astype('str')

        for i in tqdm(range(data_file.shape[0])):
            for column in data_file.columns:
                #1
                # удалим случаи типо 50х50 или 50х50х50
                data_file[column][i] = re.sub(r'\d+[xX]\d+| \d+[xX]\d+[xX]\d+', '', data_file[column][i])  # eng
                data_file[column][i] = re.sub(r'\d+[хХ]\d+| \d+[хХ]\d+[хХ]\d+', '', data_file[column][i])  # rus
                # убираем все цифры
                data_file[column][i] = re.sub(r'\d+', '', data_file[column][i])
                #2
                #для сохранения биграмм и коллокаций замена всех типов ковычек на "
                data_file[column][i] = re.sub(r'[«„“»\']', '"', data_file[column][i])
                # убираем все знаки кроме "-"
                data_file[column][i] = re.sub(r'[^a-zA-Z],[-"]', '', data_file[column][i])
                # working with words
                for word in tokenizer.tokenize(data_file[column][i]):
                    #word = re.sub(r'"', '', word)# delete brackets in bigram words
                    if (len(word) > 1) and (not word in preprocessor.stop_words_rus) and (not word in preprocessor.stop_words_eng) and (not word in preprocessor.abbreviation):
                        word = pymorphy2.MorphAnalyzer().parse(word)[0].normal_form# turn into a normal form
                        words.append(word)
        #Statistics
        #   print(words)
        #   print('Length of list "words"  with repeats: ', len(words))
        #   print('Length of list "words"  without repeats: ', len(set(words)), "\n")
        return words

    def prep_str(str):

        #SETS
        stop_words_rus = set(stopwords.words('russian'))
        stop_words_eng = set(stopwords.words('english'))

        str = re.sub(r'\d+[xX]\d+| \d+[xX]\d+[xX]\d+', '', str)  # eng
        str = re.sub(r'\d+[хХ]\d+| \d+[хХ]\d+[хХ]\d+', '', str)  # rus
        str = re.sub(r'\d+', '', str)

        str = re.sub(r'[]«[}{»()<>/+*_"-:;.,]', ' ', str).lower()
        str = re.sub(r'[@#$%^&`|!?~]', ' ', str)

        str = re.sub("[^\w]", " ", str).split()

        stop_words_rus = set(stopwords.words('russian'))
        stop_words_eng = set(stopwords.words('english'))

        words = [word for word in str if (not word in preprocessor.stop_words_rus)and(not word in preprocessor.stop_words_eng)and(not word in preprocessor.abbreviation)and(len(word)>1)]
        str = ' '.join(words)

        return str

df = pd.read_csv('data/123.csv', sep='\t', encoding='PT154')
test_str="Стол “Премьер-Элит“ разм. 80Х150х120см черн."

words = preprocessor.prep_df(df)
str = preprocessor.prep_str(test_str)
#Statistics
print(words)
#print('Length of list "words"  with repeats: ', len(words))
#print('Length of list "words"  without repeats: ', len(set(words)), "\n")

#print(str)










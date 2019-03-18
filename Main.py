import pandas as pd
import numpy as np
import re
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
<<<<<<< HEAD
import pymorphy2
import numpy as np
import re
import pymorphy2
import nltk
import nltk.tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import *
from tqdm import tqdm
from nltk.stem import *
import difflib
import codecs
from Levenshtein import *
from jellyfish import *
import distance
import textdistance
from difflib import SequenceMatcher
import pymorphy2
from pymorphy2 import MorphAnalyzer, units

def noun_selector(input_str):
    morph = pymorphy2.MorphAnalyzer()

    input_str = input_str.split()
    #print(input_str)
    bb = []
    for elem in input_str:
        p = morph.parse(elem)[0].normal_form  # ????
        p = str(morph.parse(p)[0].tag.POS)
        if (p == 'NOUN'):
            bb.append(elem)
    str12 = ' '.join(bb)
    return str12

def row_preprocessor(str):
    # sets
=======
'''
Notes:
1.не таблицу выводить, а только номера без повторов по слову или фразе
2. сделать в препроцессор запись в файл
'''
#!!!!!!!!!!
def row_preprocessor(str):

    #sets
>>>>>>> c9e3e02949980efbab3dbe359075a47f94ac9863
    stop_words_rus = set(stopwords.words('russian'))
    stop_words_eng = set(stopwords.words('english'))
    abbreviation = ["рад", "мрад", "мкрад",  # угол
                    "см", "м", "мм", "мкм", "нм", "дм",  # метр
                    "кг", "мг", "мкг", "г", "т",  # вес
                    "мин", "ч", "сут", "с", "мс", "мкс", "нс",  # время
                    "л",  # объем
                    "гц", "ггц", "мгц", "кгц",  # Гц
                    "шт",  # кол-во
                    "ом", "а", "в",  # эл-тех
                    "млн", "тыс", "млрд", "руб", "ме",  # денеж.ср.
                    "бит", "байт", "кбай", "мбайт", "гбайт", "тбайт", "мбайт"]  # информ.

    str = re.sub(r'\d+[xX]\d+| \d+[xX]\d+[xX]\d+', '', str)  # eng
    str = re.sub(r'\d+[хХ]\d+| \d+[хХ]\d+[хХ]\d+', '', str)  # rus
<<<<<<< HEAD
    str = re.sub(r'[A-z]', "", str)  # удаление английских литералов
    str = re.sub(r'\d+', '', str).lower()
    str = re.sub("[^\w]", " ", str).split()

    words = [word for word in str if
             (not word in abbreviation) and (not word in stop_words_rus) and (not word in stop_words_eng) and (
                         len(word) > 2)]
    # print(words)
=======
    str = re.sub(r'\d+', '', str).lower()
    str = re.sub("[^\w]", " ", str).split()

    words = [word for word in str if (not word in abbreviation)and(not word in stop_words_rus)and(not word in stop_words_eng)and(len(word)>2)]
>>>>>>> c9e3e02949980efbab3dbe359075a47f94ac9863

    str = ' '.join(words)
    str = noun_selector(str)
    return str


def reader(file):
    file_str = file['FullName']  # данные
    str1 = []
    for row in file_str:
        row = row_preprocessor(row)  # удаление пробелов
        str1.append(row)
    # print(str1)

    file_count = file['Count']  # данные
    str2 = []
    for row in file_count:
        str2.append(str(row))
    # print(str2)

    return str1, str2

def count_finder(file,SEARCH_WORD):
    file = file[file['Data'].str.contains(SEARCH_WORD.lower(), flags=re.I)]
    sorted_file=file
    file = file['Count']  # данные
    str3 = []
    for row in file:
        str3.append(str(row))
    return set(str3),sorted_file

<<<<<<< HEAD
def count_finder(file, SEARCH_WORD):
    file = file[file['Data'].str.contains(SEARCH_WORD.lower(), flags=re.I)]
    # sorted_file=file
    file = file['Count']  # данные
    str3 = []
    for row in file:
        str3.append(str(row))
    return set(str3)  # ,sorted_file


def saver(str1, df):
    list_of_sets = []
    for elem in str1:
        counts = count_finder(df, elem)
        list_of_sets.append(counts)

    df2 = pd.DataFrame({'Data': str1, 'List of sets': list_of_sets})
    df2.to_csv('data/data_counts.csv', sep='\t', encoding='PT154')


def jaro(df, SEARCHING_STR):
    df = df['Data']  # данные
    str3 = []
    for row in df:
        seq = difflib.SequenceMatcher(a=row.lower(), b=SEARCHING_STR.lower()).ratio()
        str3.append(seq)
    # print(str3)
    return str3


FILE = 'data/123.csv'
# FILE = 'data/nomenklatura.csv'
SEARCH_WORD = 'картридж'
SEARCHING_STR = 'принтер'

file = pd.read_csv(FILE, sep='\t', encoding='PT154')

str1, str2 = reader(file)
df = pd.DataFrame({'Count': str2, 'Data': str1})
print(df)#64869

#DROP DUPLICATES
#df = df.drop_duplicates(subset=['Data'], keep=TRUE)#remove dublicates
#print(df)#31478


#===========COUNTS->STRINGS====================
# SEARCH_COUNT = '10.12'
# df_count = df[df['Count'] == SEARCH_COUNT]
# print(df_count[10:20])
#===========SEARCH_WORD->COUNTS====================
# counts,sorted_file= count_finder(df,SEARCH_WORD)
# print(sorted_file)
# print(sorted(counts))
#===========Saver====================
# saver(str1,df)
#===========Jaro====================
#print('Jaro = ', SEARCHING_STR)
#str3 = jaro(df, SEARCHING_STR)
#df = pd.DataFrame({'Count': str2, 'Data': str1, 'Jaro': str3})
#print(df[df['Jaro'] >= 0.75])






#Rubbish
'''
#print(df)#64869
#df = df.drop_duplicates(subset=['Count', 'Data'], keep=False)#remove dublicates
#print(df)#31478

#SEARCH_COUNT = '10.12'
#df_count = df[df['Count'] == SEARCH_COUNT]
#print(df_count[10:20])

#counts,sorted_file= count_finder(df,SEARCH_WORD)
#print(sorted_file)
#print(sorted(counts))

#saver(str1,df)



file = pd.read_csv(FILE, sep='\t', encoding='PT154')

str1, str2 = reader(file)
df = pd.DataFrame({'Count': str2, 'Data': str1})
#print(df)#64869
#df = df.drop_duplicates(subset=['Count', 'Data'], keep=False)#remove dublicates
#print(df)#31478

SEARCH_COUNT = '10.12'
df_count = df[df['Count'] == SEARCH_COUNT]
#print(df_count[10:20])


counts,sorted_file = count_finder(df,SEARCH_WORD)
print(sorted_file)
print(sorted(counts))



def select_rows(df, search_strings):
    unq, IDs = np.unique(df, return_inverse=True)  # находит ункальные элементы в датасете
    unqIDs = np.searchsorted(unq, search_strings)  # находит, сортирует по search strings
    return df[((IDs.reshape(df.shape) == unqIDs[:, None, None]).any(-1)).all(0)]





str3=[]
for elements in str1:
    str3.append(elements.split())
#print(str3)

df = pd.DataFrame({"Counts": str2, "Data": str3})
search_word = 'принтер'
print(df)



#print(df1)
#dd=select_rows(df1,["пемолюкс"])
#print(dd)


#df1 = df[(if search_word in df.Data)]
#print(df1)


print(df.Data.isin(["принтер"]))

df_modified = select_rows(df,["ard"])
print(df_modified)





















test_str = "мама папа мама папа 23 принтер сканер"
search_word = "23 "
if search_word in test_str:
    print(test_str)
else:
    print(None)
=======
#FILE = 'data/123.csv'
FILE = 'data/nomenklatura.csv'
SEARCH_WORD = 'brother'

file = pd.read_csv(FILE, sep='\t', encoding='PT154')

str1, str2 = reader(file)
df = pd.DataFrame({'Count': str2, 'Data': str1})
#print(df)
>>>>>>> c9e3e02949980efbab3dbe359075a47f94ac9863

counts,sorted_file = count_finder(df,SEARCH_WORD)
print(sorted_file)
print(sorted(counts))

#сделать счета строкой

data = pd.DataFrame({"A":[" dd app app","ban", "babcy app"], "C":str2 })
print(data)


df_modified = select_rows(data,["ap"])
print(df_modified)

dict_out = {}
for x, y in zip(str2, str1):
    dict_out[x] = y
print(dict_out)


name1 = "мосгостранс"
flag = True

for search_emploers in dict_out:
    if dict_out[search_emploers]['name'] == name1:
        print(employer[search_emploers]['Telefon'])
        flag = False

if flag:
    print('нет такого значения')














lookup_list = pd.Series(dict_out)

print(lookup_list)
#print(lookup_list[lookup_list.values == 10.01])

answer = lookup_list[lookup_list.values == 10.01].index
print(answer)
answer = pd.Index.tolist(answer)
print(answer)







a=[]
def searcher (clue_word, dict_out):

    if clue_word in dict_out.values():
        a.append(dict_out.keys())
    return print(a)

#нужно написать

a = searcher(clue_word, dict_out)
#вход строка, вывод номер
#print(list(dict_out.keys())[list(dict_out.values()).index("cредство чистящее универсальное  пемолюкс  гр  ")])



for row in file:
    #row = preprocessor(row)
    str1.append(row)

print(str1)
print(len(str1))


str2 = []
file_count = file['Count']#счета
for row in file_count:
    str2.append(row)

print(str2)
print(len(str2))

uniq_and_fifa = dict(zip(str2, str1))
#print(uniq_and_fifa)


dict_out = {}

#for x, y in zip(str2, str1):
    #dict_out[x] = y
    #dict_out.update(y)
#print(dict_out)



file = pd.read_csv('data/123.csv', sep='\t', encoding='PT154')


str1 = []
file_str = file['FullName']#данные
for row in file_str:
    #row = preprocessor(row)
    str1.append(row)
    #вставим в словарь (value)

print(str1)
#print(type(str1))

str2 = []
file_count = file['Count']#счета
for row in file_count:
    str2.append(row)
    #вствим в словарь (key)

print(str2)
#print(type(str2))


pd.DataFrame(str2, columns=["BB", "aa"])    


#voc = dict(zip(str2, str1))
#print(voc)
#print(type(voc))

#bb=pd.Series(str2, str1)
#print(bb)

#for i in  str2.count():

word = "поездка"

if word in str2:
    print(ffff)
    print(str1)


answer = finder(word,str)


def finder (word, str):

    answer = [print(str[i]) for i in range(len(str)) if word in str[i]]

    return answer

берем str
очистка
бьем на words
очистка
соединяем в str


bb(str1)

def bb (str):
    words = [str]
    words = [re.sub(r'\d+', '', str)]
    #tokenizer = RegexpTokenizer('\w+-\w+|\w+')

    #str = re.sub(r'\d+', '', str)
    #words = [str]
    return words
    
    for word in tokenizer.tokenize(str_list):
        if (len(word) > 1):  # words longer 1 letters
            word = re.sub(r'"', '', word)  # delete brackets in bigram words
            word = morph.parse(word)[0].normal_form  # turn into a normal form
            words.append(word.lower())  # making all letters lowercase

        # making stop words free (rus and eng)
    stop_words_rus = set(stopwords.words('russian'))
    stop_words_eng = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words_rus | stop_words_eng]




for row in data_file:
    str1.append(row)



#str1.append(data_file)
print(str1)

def preprocessor(data_file):
    # making string type
    for column in data_file.columns:
        data_file[column] = data_file[column].astype('str')  # строчки файла data_file имеют тип String

    # token (only: word, word-word or bigrams)
    tokenizer = RegexpTokenizer('\w+-\w+|"\w+\s\w+"|\w+')  # for word and word-word and "word word" (as bigram)

    morph = pymorphy2.MorphAnalyzer()
    words = list()  # list('abc') -> ['a', 'b','c']

    # creat a list of "pure" normalised words from data_file
    # working with columns, clean from brackets, signs and numbers
    for i in tqdm(range(data_file.shape[0])):
        for column in data_file.columns:
            # убираем ковычки, скобки и знак запятой
            data_file[column][i] = re.sub(r'[]«[}{»(,)<>]', ' ', data_file[column][i])
            # удалим случаи типо 50х50 или 50х50х50
            data_file[column][i] = re.sub(r'\d+[xX]\d+| \d+[xX]\d+[xX]\d+', ' ', data_file[column][i])  # eng
            data_file[column][i] = re.sub(r'\d+[хХ]\d+| \d+[хХ]\d+[хХ]\d+', ' ', data_file[column][i])  # rus
            # убираем знаки-разделители, заменяем пробелом
            data_file[column][i] = re.sub(r'[/+*_:.]', ' ', data_file[column][i])
            # убираем все цифры
            data_file[column][i] = re.sub(r'\d+', '', data_file[column][i])
            # убираем все знаки кроме "-"
            data_file[column][i] = re.sub(r'[^a-zA-Z],[-]', '', data_file[column][i])

            # working with words
            # берет строки таблицы и делит их на составляющие (пока что он съедает тире и точки)
            for word in tokenizer.tokenize(data_file[column][i]):
                if (len(word) > 1):  # words longer 1 letters
                    word = re.sub(r'"', '', word)  # delete brackets in bigram words
                    word = morph.parse(word)[0].normal_form  # turn into a normal form
                    words.append(word.lower())  # making all letters lowercase

    # making stop words free (rus and eng)
    stop_words_rus = set(stopwords.words('russian'))
    stop_words_eng = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words_rus | stop_words_eng]

    # result
    #   print(words)
    #   print('Length of list "words"  with repeats: ', len(words))
    #   print('Length of list "words"  without repeats: ', len(set(words)), "\n")
    return words

'''

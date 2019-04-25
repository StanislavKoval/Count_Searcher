import pandas as pd
import numpy as np
import difflib
import pymorphy2
import re

from nltk.corpus import stopwords
from tqdm import tqdm

"""
Описание поставленной задачи 

На вход поступает nomenklatura.csv
    +считываем данные 
    +проверяем данные на наличие полных дублей
    +очищаем данные от шума препроцессором
    +выделяем существительные
    +формируем файл {'Count': str2, 'Data': str1}
    +проверяем данные на наличие полных дублей
    +составляем из сущ set1
    +находим в соотв сущ существующие счета
    +формируем файл {10.08, 10.09}       бита
    +находим все случаи инициализации более 1го счета set of bad words
    
    Злодеи:
    по 1му тегу из set of bad words, выводдим все строки содержащие этот тег
    Задаем вопрос: как инициализировать данную группу ? Записываем либо новый count для группы (словарь),
    либо "UNK", в данном случае все остаются при старых счетах
    все случаи с "UNK" записываем в отдельный str, а затем в файл
    
    Мирные:
    все строки nomenklatura, входящие в list of good str остаются при своих счетах
    
    совмещаем получившиеся новый после переинициализации список и мирный
    сохраняем файл3 как nomenklatura_new (fuck yeah!!)

На выход поступает nomenklatura_new.csv и файл с "UNK"

"""


def noun_selector_for_str(input_str):
    morph = pymorphy2.MorphAnalyzer()
    input_str = input_str.split()
    bb = []
    for elem in input_str:
        # p = morph.parse(elem)[0].normal_form#????
        p = str(morph.parse(elem)[0].tag.POS)
        if (p == 'NOUN'):
            bb.append(elem)
    str3 = ' '.join(bb)
    return str3


def row_preprocessor(str):
    # sets
    stop_words_rus = set(stopwords.words('russian'))
    # stop_words_eng = set(stopwords.words('english'))
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
    str = re.sub(r'[A-z]', "", str)  # удаление английских литералов
    str = re.sub(r'\d+', '', str).lower()
    str = re.sub("[^\w]", " ", str).split()

    words = [word for word in str if  # (not word in stop_words_eng) and
             (not word in abbreviation) and (not word in stop_words_rus) and (len(word) > 3)]

    str = ' '.join(words)  # получаем ощищенную от шума строку
    return str


def reader(file):
    file_str = file['Fullname']
    str1 = []
    for row in file_str:
        row = row_preprocessor(row)  # очищаем данные от шума
        if (row.find(' ') != -1):  # условие в котором мы сохраняем строки с 1м словом
            row = noun_selector_for_str(row)  # выделяем существительные
        str1.append(row)
    # print(str1)

    file_count = file['Count']  # данные
    str2 = []
    for row in file_count:
        str2.append(str(row))
    # print(str2)

    return str1, str2


def count_finder(file, SEARCH_WORD, column_name):
    str3 = []
    file = file[file[column_name].str.contains(SEARCH_WORD.lower(), flags=re.I)]
    # sorted_file = file
    file = file['Count']  # данные
    for row in file:
        str3.append(str(row))
    str4 = sorted(set(str3))

    return str4  # , sorted_file


def saver(str1, df):
    list_of_sets = []
    for elem in str1:
        counts = count_finder(df, elem, 'Data')
        list_of_sets.append(counts)

    df2 = pd.DataFrame({'Data': str1, 'List of sets': list_of_sets})
    df2.to_csv('data/data_counts.csv', sep='\t', encoding='PT154')


def pure_saver(df, name_of_file):  # 'data/data_counts12.csv'
    return df.to_csv(name_of_file, sep='\t', encoding='PT154')  # сделат во входных данных еще и имя для сохранения


def jaro(df, SEARCHING_STR):
    df = df['Data']  # данные
    str3 = []
    for row in df:
        seq = difflib.SequenceMatcher(a=row.lower(), b=SEARCHING_STR.lower()).ratio()
        str3.append(seq)
    # print(str3)
    return str3


def teg_to_counts(str1, df):
    one_str, set_of_counts = [], []
    set_of_tegs = set()
    for elem in str1:
        one_str = elem.split()
        for word in one_str:
            set_of_tegs.add(word)  # set тегов
    set_of_tegs = sorted(set_of_tegs)  # отсортированный по алфавиту set тегов без повтора

    for word in set_of_tegs:
        counts_of_word = count_finder(file=df, SEARCH_WORD=word, column_name='Data')
        set_of_counts.append(counts_of_word)

    # запишем данные в файл #как дополнение можно записать данные в dict
    # {10.08, 10.09} бита
    # {10.05}        блок
    df_teg_to_counts = pd.DataFrame({'TEG': set_of_tegs, 'Set of counts': set_of_counts})

    # создадим отдельный файл со всеми
    set_bad_words = set()
    for word in set_of_tegs:
        counts_of_word = count_finder(file=df, SEARCH_WORD=word, column_name='Data')
        if (len(counts_of_word) != 1):
            # print('Word:', word,'\t', 'Counts:', counts_of_word,'\n')
            set_bad_words.add(word)
    set_bad_words = sorted(set_bad_words)
    return df_teg_to_counts, set_bad_words


def str_finder(file, set_bad_words):  # на вход nomenklatura

    str4 = set()
    list_of_bad_str = set()

    file_str = file['Fullname']
    for row in file_str:
        str4.add(row)
        row_low = row.lower()
        for i in set_bad_words:
            if (row_low.find(i) != -1):
                list_of_bad_str.add(row)

    list_of_good_str = str4 - list_of_bad_str
    list_of_bad_str = sorted(list_of_bad_str)
    list_of_good_str = sorted(list_of_good_str)
    # print(str4)
    # print(list_of_bad_str)
    # print(list_of_good_str)
    return list_of_bad_str, list_of_good_str


#########################################################################################################
FILE = 'data/123.csv'
# FILE = 'data/nomenklatura.csv'

# считываем файл
file = pd.read_csv(FILE, encoding='PT154')

# удаляет полные дубли по 2м столбцам
file_1 = file.drop_duplicates(subset=None, keep='first')
# print(file_1)
# Cчитываем столбцы
str1, str2 = reader(file_1)
# print("Str1:",str1[:2])# str1 ['cущ1 сущ2 сущ3','сущ1 сущ2 сущ3',...]
# print("Str2:",str2[:2])# str2 [Count1,Count2,...]

# Формируем новый файл file_2 {'Count': str2, 'Data': str1}
file_2 = pd.DataFrame({'Count': str2, 'Data': str1})
# удаляет полные дубли по 2м столбцам
file_2 = file_2.drop_duplicates(subset=None, keep='first')
# print(file_2)

# Формируем новый файл file_3 {'Set of counts':[10.08, 10.09], 'TEG':'бита'}
file_3, set_bad_tegs = teg_to_counts(str1=str1, df=file_2)
# print(file_3)
#print(set_bad_tegs)

# Сохраним наработки по file3
pure_saver(df=file_3, name_of_file='data/df_teg_to_counts.csv')

# Нашли все строки содержащие и не содержащие set_bad_tegs
list_of_bad_str, list_of_good_str = str_finder(file_1, set_bad_tegs)

# Создадим [['Count','Data'],['Count','Data'],[]] для good c исходными значениями
df = file_1.values
good_elems = []
for elem in df:
    for i in elem:
        if i in list_of_good_str:
            elem = list(elem)
            good_elems.append(elem)
            # print(elem)


# print(good_elems)#[['10.01', '90 минут 2 поездка  17.12.14 МосГосТранс'], ['10.12', 'Средство чистящее универсальное


def reclass_bad_str(set_bad_tegs, list_of_bad_str, str2):
    cc = list()
    bbbb = list()
    qqqq = list()
    jj = list()
    print(set(str2))
    for word in set_bad_tegs:  # бита
        bb = list()
        for elem in list_of_bad_str:
            elem_low = elem.lower()
            if (elem_low.find(word) != -1):
                bb.append(elem)
        # print(word,bb)#бита ['БИТА',...]

        print('Введите номер счета следующей группе строк с TEGом -', word, '\n')
        for i in bb:
            print(' ', i)
        input_count = input('\nВведите номер счета:')
        print('Вы ввели', input_count, '\n')
        input_count = str(input_count)


        if input_count in set(str2):
            cc.append(input_count)
        else:
            input_count = 'UNK'
            cc.append(input_count)
        bbbb.append(bb)

        for i in bb:
            qq = [input_count]
            qq.append(i)
            qqqq.append(qq)
    # print(cc)
    # jj=list(zip(cc, bbbb))
    # print(list(zip(cc,bbbb)))#[('12', ['БИТА', 'Бита', 'Бита 1/4"', 'Бита 100мм', 'Бита ЗУБР МАСТЕР с 1/4 НР 2']),
    print(type(qqqq))

    #удалим дубли в qqqq

    return qqqq


# нужно доработать qqqq на наличие одинаковых элементов, которые были инициализированы по разному


# vocab, vocab_1=reclass_bad_str(set_bad_tegs)
qqqq = reclass_bad_str(set_bad_tegs, list_of_bad_str,str2)

# print(vocab)


ultra = qqqq + good_elems


for i in ultra:
    print(' ', i)

print(type(ultra))
print(len(ultra))

'''
1)
если аналитик не знает как классифицировать, 
или код не удовлетворяет формату счета из словаря счетов, 
ставь UNK

2)
preprocessor
исследовать три первых слова для тега 

3)
Расстояние левинштейна, 
доп проверка, были ли использованы в классификации следующие строки

'''

# Rubbish
'''
def reclass_bad_str(set_bad_tegs,list_of_bad_str):
    cc=list()
    bbbb=list()
    qqqq=list()
    jj=list()
    for word in set_bad_tegs:#бита
        bb=list()
        for elem in list_of_bad_str:
            elem_low=elem.lower()
            if (elem_low.find(word) != -1):
                bb.append(elem)
        #print(word,bb)#бита ['БИТА',...]


        print('Введите номер счета следующей группе строк с TEGом -', elem, '\n')
        for i in bb:
            print(' ', i)
        input_count = input('\nВведите номер счета:')
        print('Вы ввели', input_count, '\n')
        input_count=str(input_count)
        cc.append(input_count)
        bbbb.append(bb)

        for i in bb:
            qq=[input_count]
            qq.append(i)
            qqqq.append(qq)
    #print(cc)
    #jj=list(zip(cc, bbbb))
    #print(list(zip(cc,bbbb)))#[('12', ['БИТА', 'Бита', 'Бита 1/4"', 'Бита 100мм', 'Бита ЗУБР МАСТЕР с 1/4 НР 2']),
    #print(qqqq)
    return qqqq


buffer = list()
        for i in list_of_bad_str:
            i_low = i.lower()
            if (i_low.find(word) != -1):
                buffer.append(i)
            vocab[word] = buffer
            
    #vocab_1 = vocab  # слово и все строки исходного файла содержащие bad word #х

   
    for elem in vocab:
        bb = []
        print('Введите номер счета следующей группе строк с TEGом -', elem, '\n')
        for i in vocab[elem]:
            print(' ', i)
        input_count = input('\nВведите номер счета:')
        print('Вы ввели', input_count, '\n')
        input_count = str(input_count)
        vocab[input_count] = vocab.pop(elem)

        bb.append(input_count)
        bb.append(vocab[input_count])
        bbbb.append(bb)
    #print(bbbb)
    return vocab, vocab_1









df = file_1.values
good_elems = []
for elem in df:
    for i in elem:
        if i in list_of_good_str:
            elem = list(elem)
            good_elems.append(elem)
            #print(elem)



# создадим словарь слово:строки где встречаются

def reclass_bad_str(set_bad_tegs):
    vocab = dict()

    bbbb=[]

    for word in set_bad_tegs:
        buffer = set()
        for i in list_of_bad_str:
            i_low = i.lower()
            if (i_low.find(word) != -1):
                buffer.add(i)
            vocab[word] = buffer
    vocab_1 = vocab#слово и все строки исходного файла содержащие bad word


    for elem in vocab:
        bb = []
        print('Введите номер счета следующей группе строк с TEGом -', elem, '\n')
        for i in vocab[elem]:
            print(' ', i)
        input_count = input('\nВведите номер счета:')
        print('Вы ввели', input_count, '\n')
        input_count = str(input_count)
        vocab[input_count] = vocab.pop(elem)

        bb.append(input_count)
        bb.append(elem)
    bbbb.append(bb)
    print(bbbb)
    return vocab, vocab_1


vocab, vocab_1 = reclass_bad_str(set_bad_tegs)
#new_dict = list(zip(vocab_count, vocab_value))

#print(vocab)#{'12': {'Бита 100мм', 'БИТА', 'Бита 1/4"', 'Бита', 'Бита ЗУБР МАСТЕР с 1/4 НР 2'}, '13': {'Стол “Премьер-Элит“ разм. 80Х150х120см черн.', 'Костюм охрана-с размер 56-58  рост 182-188   ( депо Баумана)', 'Перчатки желтые резиновые размер XL '}, '14': {'Костюм охрана-с размер 56-58  рост 182-188   ( депо Баумана)', 'Перчатки желтые резиновые размер XL '}}
#vocab={'12': {'Бита 100мм', 'БИТА', 'Бита 1/4"', 'Бита', 'Бита ЗУБР МАСТЕР с 1/4 НР 2'}, '13': {'Стол “Премьер-Элит“ разм. 80Х150х120см черн.', 'Костюм охрана-с размер 56-58  рост 182-188   ( депо Баумана)', 'Перчатки желтые резиновые размер XL '}, '14': {'Костюм охрана-с размер 56-58  рост 182-188   ( депо Баумана)', 'Перчатки желтые резиновые размер XL '}}


good_elems=[]
bad_elems=[]
df = file_1.values
for elem in df:
    for i in elem:
        if i in list_of_good_str:
            good_elems.append(elem)
        else:
            bad_elems.append(elem)
#print(good_elems)





for elem in bb:
    for i in elem:
        if (i%2==0):
            for j in i:


print(bb)

# создать словарь с правильными словами (все - принадлежащие к неправильным)

file_str = file_1['Fullname']
vocab_value = []
for row in file_str:
    vocab_value.append(row)
#print(vocab_value)

file_count = file_1['Count']
vocab_count = []
for row in file_count:
  vocab_count.append(row)
#print(vocab_count)

new_dict = list(zip(vocab_count, vocab_value))
for elem in new_dict:
    for i in elem:
        if (i==)
















print(len(vocab_of_all))
print(tabulate(vocab_of_all.items(), headers=['Count', 'Strings'], tablefmt="grid"))



def bad_str_finder(set_bad_words,file):  # на вход nomenklatura

    list_of_str = []
    file_str = file['Fullname']
    for row in file_str:
        list_of_str.append(row)

    print(list_of_str)
    for i in list_of_str:
        for elem in set_bad_words:
            if(i.find(elem)!=-1):
                print(i,elem)

    return
bad_str_finder(set_bad_words=set_bad_words,file=file_1)



# Формируем списки. list_of_bad_str слова которые нужно переквалифицировать
vocabulary_1 = bad_str_finder(file_1, set_bad_words)
#list_of_bad_str, list_of_good_str,
#{Сount:['','',''],Сount:['','',''],...}

print(vocabulary_1)

def bad_str_finder(file, set_bad_words):  # на вход nomenklatura

    str4 = set()
    list_of_bad_str = set()

    #{Count from input:['str1','str2','str3', ...]}
    vocabulary_1=dict()
    vocabulary_UNK=dict()
    list_of_bad_str_UNK=[]

    file_str = file['Fullname']
    for row in file_str:
        str4.add(row)
        row_low = row.lower()
        for i in set_bad_words:
            print(i)
            if (row_low.find(i) != -1):
                list_of_bad_str.add(row)
                print(list_of_bad_str)
               
            print("Введите номер счета для строк по слову - ",i)
            print(list_of_bad_str)
            input_count = str(input())
            print('Вы ввели:',input_count)
            if(input_count=="UNK"):
                print("Cтроки инициализированны как 'UNK'")
                vocabulary_UNK[input_count] = list_of_bad_str
                list_of_bad_str_UNK.append(row)
            print("Группа строк инициализирована по номеру -  ", input_count)
            vocabulary_1[input_count] = list_of_bad_str
            

    #list_of_good_str = str4 - list_of_bad_str
    #list_of_bad_str = sorted(list_of_bad_str)
    #list_of_good_str = sorted(list_of_good_str)

    #return vocabulary_1  #list_of_bad_str, list_of_good_str,

empty_list = np.zeros(len(list_of_bad_str))
file_for_classification = pd.DataFrame({'Count': empty_list, 'Data': list_of_bad_str})
print(file_for_classification)


#file_12 = file_1[file_1['Fullname'] == (elem for elem in list_of_good_str)]
#print(file_12)




empty_list = np.zeros(len(list_of_bad_str))
file_for_classification = pd.DataFrame({'Count': empty_list, 'Data': list_of_bad_str})
print(file_for_classification)


#file_12 = file_1[file_1['Fullname'] == (elem for elem in list_of_good_str)]
#print(file_12)





file_12 = file_1[file_1[] == (elem for elem in list_of_good_str)]
print(file_12)



#file.loc[len(file)] = [file_2['Data'].str.contains(word, flags=re.I)]


#str1, str2 = reader(file)
#df = pd.DataFrame({'Count': str2, 'Data': str1})
#print(df)#64869



# ===========COUNTS->STRINGS====================
# SEARCH_COUNT = '10.12'
# df_count = df[df['Count'] == SEARCH_COUNT]
# print(df_count[10:20])
# ===========SEARCH_WORD->COUNTS====================
# counts,sorted_file= count_finder(df,SEARCH_WORD)
# print(sorted_file)
# print(sorted(counts))
# ===========Saver====================
# saver(str1,df)
# ===========Jaro====================
# print('Jaro = ', SEARCHING_STR)
# str3 = jaro(df, SEARCHING_STR)
# df = pd.DataFrame({'Count': str2, 'Data': str1, 'Jaro': str3})
# print(df[df['Jaro'] >= 0.75])
# ===========Noun->Counts====================
#df_presentation = nouns_to_counts(str1)
#print(df_presentation)
#pure_saver(df_presentation)
#Statistic
#9532 всего строк
#5304 проклассифицированы более 1 раза
#2)частота упоминаний в документе сущностей
#СОХРАНИ ФАЙЛ+
#без лемманизации
#9554 всего строк
#5320 проклассифицированы более 1 раза














from nltk.collocations import *
from tqdm import tqdm
from nltk.stem import *
import numpy as np
from pymorphy2 import MorphAnalyzer, units
import codecs
import distance
import textdistance
from difflib import SequenceMatcher
import re
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import pymorphy2
import numpy as np
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
#from Levenshtein import *
#from jellyfish import *

import pandas as pd
import re
from nltk.corpus import stopwords
import difflib
import pymorphy2


def noun_selector_for_str(input_str):
    morph = pymorphy2.MorphAnalyzer()
    input_str = input_str.split()
    bb=[]
    for elem in input_str:
        #p = morph.parse(elem)[0].normal_form#????
        p = str(morph.parse(elem)[0].tag.POS)
        if (p =='NOUN'):
            bb.append(elem)
    str3 = ' '.join(bb)
    return str3

def row_preprocessor(str):
    # sets
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
    str = re.sub(r'[A-z]', "", str)  # удаление английских литералов
    str = re.sub(r'\d+', '', str).lower()
    str = re.sub("[^\w]", " ", str).split()

    words = [word for word in str if
             (not word in abbreviation) and (not word in stop_words_rus) and (not word in stop_words_eng) and (
                     len(word) > 2)]
    # print(words)

    str = ' '.join(words)
    str = noun_selector_for_str(str)
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

def count_finder(file, SEARCH_WORD):
    file = file[file['Data'].str.contains(SEARCH_WORD.lower(), flags=re.I)]
    #sorted_file = file
    file = file['Count']  # данные
    str3 = []
    for row in file:
        str3.append(str(row))
    return set(str3)#, sorted_file

def saver(str1, df):
    list_of_sets = []
    for elem in str1:
        counts = count_finder(df, elem)
        list_of_sets.append(counts)

    df2 = pd.DataFrame({'Data': str1, 'List of sets': list_of_sets})
    df2.to_csv('data/data_counts.csv', sep='\t', encoding='PT154')

def pure_saver(df):
    return df.to_csv('data/data_counts12.csv', sep='\t', encoding='PT154')#сделат во входных данных еще и имя для сохранения

def jaro(df, SEARCHING_STR):
    df = df['Data']  # данные
    str3 = []
    for row in df:
        seq = difflib.SequenceMatcher(a=row.lower(), b=SEARCHING_STR.lower()).ratio()
        str3.append(seq)
    # print(str3)
    return str3

def nouns_to_counts(str1):
    kk, mm, cube = [], [], []
    for elem in str1:
        kk = elem.split()
        for word in kk:
            mm.append(word)
    mm = list(set(mm))

    for word in mm:
        counts = count_finder(df, word)
        cube.append(counts)

    df_presentation = pd.DataFrame({'NOUN': mm, 'COUNTS': cube})

    return df_presentation


FILE = 'data/123.csv'
#FILE = 'data/nomenklatura.csv'

#SEARCH_WORD = 'картридж'
#SEARCHING_STR = 'принтер на продажу '

file = pd.read_csv(FILE, sep='\t', encoding='PT154')

#str1, str2 = reader(file)
#df = pd.DataFrame({'Count': str2, 'Data': str1})
#print(df)#64869

# DROP DUPLICATES
# df = df.drop_duplicates(subset=['Data'], keep=TRUE)#remove dublicates
# print(df)#31478


# ===========COUNTS->STRINGS====================
# SEARCH_COUNT = '10.12'
# df_count = df[df['Count'] == SEARCH_COUNT]
# print(df_count[10:20])
# ===========SEARCH_WORD->COUNTS====================
# counts,sorted_file= count_finder(df,SEARCH_WORD)
# print(sorted_file)
# print(sorted(counts))
# ===========Saver====================
# saver(str1,df)
# ===========Jaro====================
# print('Jaro = ', SEARCHING_STR)
# str3 = jaro(df, SEARCHING_STR)
# df = pd.DataFrame({'Count': str2, 'Data': str1, 'Jaro': str3})
# print(df[df['Jaro'] >= 0.75])
# ===========Noun->Counts====================
#df_presentation = nouns_to_counts(str1)
#print(df_presentation)
#pure_saver(df_presentation)
#Statistic
#9532 всего строк
#5304 проклассифицированы более 1 раза
#2)частота упоминаний в документе сущностей
#СОХРАНИ ФАЙЛ+
#без лемманизации
#9554 всего строк
#5320 проклассифицированы более 1 раза














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

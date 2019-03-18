#Библиотеки
'''
import pandas as pd
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
'''
# Задачи
'''
1)+
pymorph
nltk
pos-tegger, оставить существительные 

1.2)
если экземпляр в 1м числе оставить 
    взять очищенный файл
    указать частоты появлений в nomenklatura.csv
    строки в единственном числе остаются

2)+
подготовить алгоритм фильтрации по схожести по метрикам Джаро-Винклера
***убрать символы, неестественного языка в препроцессоре +
****убрать в выходном файле повторы

Задача:
список очишенных даных // лист (сет) со счетами

расстояние Ливенштейна ??
посимвольное отличие двух строк

механизмы поиска через расстояния


1. Нужно посмотреть у какого элемента какие счета после проведения процедуры предобработки
    например вводишь счет,а тебе выдается те строки, которые к нему относятся

2. Нужно понять какие элементы правильно/неправильн отнесены по счетам, это лучше консультироваться у Кулакова
    пн
3. Сделать так чтобы элементы были только на своем (правильном) счету


постегер вычленить сущности (набор плоских сверл дереву -> сверла)

'''
# Метрики расстояний
'''
р-е Хэмминга >> число несовпадающих символов
р-е Ливенштейна >> минимальное количество операций 
                   вставки одного символа, удаления одного 
                   символа и замены одного символа на другой, 
                   необходимых для превращения одной строки в другую
Сходство Джаро — Винклера >> Чем меньше расстояние для двух строк, 
                             тем больше сходства имеют эти строки друг с другом

textdistance                            
https://github.com/orsinium/textdistance  


# расчет посимвольной разницы, р-е Левенштейна
def distance(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)  # Keep current and previous row, not entire matrix
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]


a = 'brother 12'
b = 'факс brother 12'
bbb = 'гулять мяч красивый изготовлено человек-паук сильно и не как что потому'

#Затестить
# Джаро — Винклера
seq = difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()
print(seq)

# р-е Левенштейна
c = 0
c = distance(a, b)
print(c)
'''
# Отработка pos-tagger nltk
'''
import nltk
from nltk.tag import pos_tag_sents

#nltk.download('all')
sentence = "литература учебно методическая консультант вопросам безопасности перевозки опасных грузов"

tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged)

bb,cc=[],[]
for elem in tagged:
    for i in elem:
        if (i=='NN')or(i=='NNP')or(i=='JJ'):#проблема в инициализации первого слова
            bb.append(list(elem))
for elem in bb:
    for i in elem:
        if (i != 'NN') and (i != 'NNP') and (i != 'JJ'):
            cc.append(i)
            str = ' '.join(cc)

print("----------------------")
print(bb)

print(str)
'''
# Отработка pos-tagger pymorph
'''
import pymorphy2
from pymorphy2 import MorphAnalyzer, units


morph = pymorphy2.MorphAnalyzer()

input_str = "Картридж черный матовый модель СК521.19/21 15 мм"
print(input_str)
input_str = input_str.split()
print(input_str)
bb=[]
for elem in input_str:
    p = morph.parse(elem)[0].normal_form#????
    p = str(morph.parse(p)[0].tag.POS)
    if (p =='NOUN'):
        bb.append(elem)
str = ' '.join(bb)
print(str)
'''

import pymorphy2
from pymorphy2 import MorphAnalyzer, units

def noun_selector(input_str):
    morph = pymorphy2.MorphAnalyzer()


    input_str = input_str.split()
    print(input_str)
    bb=[]
    for elem in input_str:
        p = morph.parse(elem)[0].normal_form#????
        p = str(morph.parse(p)[0].tag.POS)
        if (p =='NOUN'):
            bb.append(elem)
    str = ' '.join(bb)
    return str









#Rubbish
'''
str_test = ['картридж черный матовый модель СК521.19/21 15 мм', 'перон ворона кактус красивый красный глубоко']
def noun_selector(str_test):

    morph = pymorphy2.MorphAnalyzer()

    bb,cc = [],[]
    for elem in str_test:
        elem = elem.split()
        #print(elem)
        for i in elem:
            p = morph.parse(i)[0].normal_form  # ????
            p = str(morph.parse(p)[0].tag.POS)
            #print(p)

            if (p == 'NOUN'):
                print(i)
                bb.append(i)
            p = 0

    #cc.append(bb)
        #str22 = ' '.join(bb)
    print(cc)




srt1 = noun_selector(str_test)
#print(str1)

str_test = ['картридж черный матовый модель СК521.19/21 15 мм', 'перон ворона кактус красивый красный глубоко']
def noun_selector(str_test):

    morph = pymorphy2.MorphAnalyzer()

    bb,cc = [],[]
    for elem in str_test:
        elem = elem.split()
        #print(elem)
        for i in elem:
            p = morph.parse(i)[0].normal_form  # ????
            p = str(morph.parse(p)[0].tag.POS)
            #print(p)

            if (p == 'NOUN'):
                print(i)
                bb.append(i)
            p = 0

    #cc.append(bb)
        #str22 = ' '.join(bb)
    print(cc)




srt1 = noun_selector(str_test)
#print(str1)

#print("--------------------")


#p = morph.parse("мыла")[0].tag.POS
#print(p)

#if (p=="ADVB"):
#    print("Это блин работает ")






p=morph.parse('сталь')[0]
p = str(p.tag.POS)
print(p)

for elem in tagged:
    for i in elem:
        if (i=='NN')or(i=='NNP')or(i=='JJ'):#проблема в инициализации первого слова
            bb.append(list(elem))
for elem in bb:
    for i in elem:
        if (i != 'NN') and (i != 'NNP') and (i != 'JJ'):
            cc.append(i)
            str = ' '.join(cc)

print("----------------------")
print(bb)#[(,),(,)]->[,,]

print(str)










import pymorphy

info = morph.get_graminfo(u"малой")
print(info[0]['norm'])  # нормальная форма

print(info[0]['class'])  # часть речи, С = существительное

print(info[0]['info'])  # род, число, падеж и т.д.


aa = a.split()
# bb = split(b)
print(aa)
sor = 1 - distance.sorensen(aa, aa)





def spellcheck(self, sentence):
    #return ' '.join([difflib.get_close_matches(word, wordlist,1 , 0)[0] for word in sentence.split()])
    return ' '.join( [ sorted( { Levenshtein.ratio(x, word):x for x in wordlist }.items(), reverse=True)[0][1] for word in sentence.split() ] )



import codecs, difflib, Levenshtein, distance

with codecs.open("titles.tsv","r","utf-8") as f:
    title_list = f.read().split("\n")[:-1]

    for row in title_list:

        sr      = row.lower().split("\t")

        diffl   = difflib.SequenceMatcher(None, sr[3], sr[4]).ratio()
        lev     = Levenshtein.ratio(sr[3], sr[4]) 
        sor     = 1 - distance.sorensen(sr[3], sr[4])
        jac     = 1 - distance.jaccard(sr[3], sr[4])

        print diffl, lev, sor, jac


#diffl = difflib.SequenceMatcher(None, a, b).ratio()
#print(diffl)

#sor = 1 - distance.sorensen("dis","car")
#print(soc)
#jac = 1 - distance.jaccard(a, b)



#print(diffl,sor,jac)

#lev = Levenshtein.ratio(a, b)
#print(lev)

print(jellyfish.levenshtein_distance(a, b))
jellyfish.jaro_distance(u'jellyfish', u'smellyfish')
jellyfish.damerau_levenshtein_distance(u'jellyfish', u'jellyfihs')

def get_bigrams(myString):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    stemmer = PorterStemmer()
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    print(bigram_finder)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)

    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)

    result = [' '.join([stemmer.stem(w).lower() for w in x.split()]) for x in tokens if x.lower() not in stopwords.words('english') and len(x) > 8]
    return result

str = "Пансионат 'Ананасовый изумруд' приветствует 'Газпром Холдинг Москва'"

str_changes = get_bigrams(str)
print("_____________________________")
print(str_changes)
print(pymorphy2.MorphAnalyzer().parse("размерная")[0].normal_form)


import nltk

from nltk import word_tokenize 

from nltk.util import ngrams


text = ['cant railway station','citadel hotel',' police stn']
for line in text:
    token =nltk.word_tokenize(line)
    bigram = list(ngrams(token,2)) 

    # the 2 represents bigram...you can change it for as many as you want 
'''

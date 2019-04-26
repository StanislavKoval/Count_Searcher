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
        one_str = elem.split()  # list [word0, word1, word2, ...]
        # для словаря тегов используем только первые 3 слова строки

        for word in one_str[:2]:  # возьмем для словаря тегов только первые 2 слова
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


def good_finder(file, list_of_good_str):
    df = file.values
    good_elems = []
    for elem in df:
        for i in elem:
            if i in list_of_good_str:
                elem = list(elem)
                good_elems.append(elem)
                # print(elem)

    # [['10.01', '90 минут 2 поездка  17.12.14 МосГосТранс'], ['10.12', 'Средство чистящее универсальное
    # print(good_elems)
    return good_elems


def reclass_bad_str(set_bad_tegs, list_of_bad_str, list_of_counts):
    list_of_new_counts = list()
    bbbb = list()
    bad_elems = list()
    # print(set(str2))

    # для каждого слова из set_bad_tegs, найдем те строки, в которых он был зафиксирован
    for word in set_bad_tegs:

        word_bad_str = list()
        for elem in list_of_bad_str:
            elem_low = elem.lower()
            if (elem_low.find(word) != -1):
                word_bad_str.append(elem)
                # ['Bad_str_0','Bad_str_1',...]

        #можно реализовать условие: Если 1 строка, то 'UNK', в противном случае на классификацию попадает все
        if len(word_bad_str) > 0:

            # Ручной ввод
            print('Введите номер счета следующей группе строк с TEGом -', word, '\n')
            for i in word_bad_str:
                print(' ', i)
            input_count = input('\nВведите номер счета:')
            print('Вы ввели', input_count, '\n')
            input_count = str(input_count)

        else:
            input_count = 'UNK'

        # Условие для UNK
        if input_count in set(list_of_counts):
            list_of_new_counts.append(input_count)
        else:
            input_count = 'UNK'
            list_of_new_counts.append(input_count)
        # bbbb.append(word_bad_str)

        # bad_elems = [('input_count', ['БИТА', 'Бита', 'Бита 1/4"', 'Бита 100мм', 'Бита ЗУБР МАСТЕР с 1/4 НР 2']),
        for i in word_bad_str:
            one_elem = [input_count]
            one_elem.append(i)
            bad_elems.append(one_elem)

    return bad_elems


def all_unk_to_file(new_data_list):
    unk_elems = []
    for elem in new_data_list:
        for i in elem:
            if i == "UNK":
                elem = list(elem)
                unk_elems.append(elem)

    unk_elems = np.array(unk_elems)
    new_list_of_counts = unk_elems[:, [0]].ravel()
    new_list_of_fullname = unk_elems[:, [1]].ravel()

    file_5 = pd.DataFrame({'Count': new_list_of_counts, 'Fullname': new_list_of_fullname})

    file_5 = file_5.drop_duplicates(subset=None, keep='first')
    file_5 = file_5.drop_duplicates(subset=['Fullname'], keep='first')
    return file_5


# в случае если bad word содержится только в 1 предложении, сохранять старый

#########################################################################################################
FILE = 'data/123.csv'
# FILE = 'data/nomenklatura.csv'

# Считываем файл
file = pd.read_csv(FILE, encoding='PT154')

# Удаляет полные дубли по 2м столбцам
file_1 = file.drop_duplicates(subset=None, keep='first')
# print(file_1)

# Считываем столбцы
str1, str2 = reader(file_1)
# print("Str1:",str1[:2])# str1 ['cущ1 сущ2 сущ3','сущ1 сущ2 сущ3',...]
# print("Str2:",str2[:2])# str2 [Count1,Count2,...]

# Формируем новый файл file_2 {'Count': str2, 'Data': str1}
file_2 = pd.DataFrame({'Count': str2, 'Data': str1})

# Удаляет полные дубли по 2м столбцам
file_2 = file_2.drop_duplicates(subset=None, keep='first')
# print(file_2)

# Формируем новый файл file_3 {'Set of counts':[10.08, 10.09], 'TEG':'бита'}
file_3, set_bad_tegs = teg_to_counts(str1=str1, df=file_2)
# print(file_3)
# print(set_bad_tegs)

# Сохраним наработки по file_3
pure_saver(df=file_3, name_of_file='data/df_teg_to_counts12.csv')

# Нашли все строки содержащие и не содержащие set_bad_tegs
list_of_bad_str, list_of_good_str = str_finder(file=file_1, set_bad_words=set_bad_tegs)

# Создадим [['Count','Data'],['Count','Data'],[]] для good c исходными значениями
good_elems = good_finder(file=file_1, list_of_good_str=list_of_good_str)
# [['10.01', '90 минут 2 поездка  17.12.14 МосГосТранс'], ['10.12', 'Средство чистящее универсальное

# Создадим [['Count','Data'],['Count','Data'],[]] для bad cо значениями введенными с клавиатуры
bad_elems = reclass_bad_str(set_bad_tegs=set_bad_tegs, list_of_bad_str=list_of_bad_str, list_of_counts=str2)
# [['10.10', 'Бита ЗУБР МАСТЕР с 1/4 НР 2'],['UNK', 'Костюм охрана-с размер 56-58  рост 182-188   ( депо Баумана)']

# Cложим 2 получившихся списков, для восстановления обновленного строя данных
new_data_list = bad_elems + good_elems
#new_data_list=[['10.10', 'БИТА'], ['10.10', 'Бита'], ['10.10', 'Бита 1/4"'], ['10.10', 'Бита 100мм'], ['10.10', 'Бита ЗУБР МАСТЕР с 1/4 НР 2'], ['UNK', 'Костюм охрана-с размер 56-58  рост 182-188   ( депо Баумана)'], ['UNK', 'Перчатки желтые резиновые размер XL '],['10.01', 'Перчатки желтые резиновые размер XL '], ['10.01', '90 минут 2 поездка  17.12.14 МосГосТранс'], ['10.12', 'Средство чистящее универсальное (Пемолюкс 480 г)'], ['10.113', 'Принтер сканер Brother XXXX'], ['10.12', 'Стол “Премьер-Элит“ разм. 80Х150х120см черн.'], ['10.11.18', 'Путевка „"Golden CARD"“ пансионат «"Ананасовый изумруд"»  "ВАО Москва"'], ['10.12', 'деталь шайба завод «Ракетно космическая корпорация»'], ['10.05', 'Утяжелитель конуса КС-27'], ['10.05', 'Ф/диск KingSton DT101G2/32GB'], ['10.05', 'Фара (блок фары) 30.3775.000 правая МАЗ Евро'], ['10.05', 'Фара (блок фары) левая МАЗ Евро'], ['10.05', 'Фара (фонарь)противотуман.'], ['10.05', 'Фара б/у'], ['10.05', 'Фара в сборе'], ['10.05', 'Фара Волжанин ближнего света'], ['10.05', 'Фара Волжанин'], ['10.05', 'Фара Г-3102'], ['10.09', 'Билет входной'], ['10.09', 'Биотуалет']]

# Преобразуем вложенный список в 2 списка: список счетов, спиок обновленный данных
new_data_list = np.array(new_data_list)
new_list_of_counts = new_data_list[:, [0]].ravel()
new_list_of_fullname = new_data_list[:, [1]].ravel()

# Формируем новый файл file_4
file_4 = pd.DataFrame({'Count': new_list_of_counts, 'Fullname': new_list_of_fullname})

# Удаляет полные дубли по 2м столбцам
file_4 = file_4.drop_duplicates(subset=None, keep='first')
file_4 = file_4.drop_duplicates(subset=['Fullname'], keep='first')

# Можно отсортировать по алфавиту
file_4 = file_4.sort_values(['Fullname', 'Count'], ascending=[True, False])
print(file_4)

# Сохраним наработки по file_4
pure_saver(df=file_4, name_of_file='data/final_for_test.csv')

# Реализуем вывод все элементов с "UNK" в отдельный файл
file_5_UNK = all_unk_to_file(new_data_list=new_data_list)
#print(file_5_UNK)


#Сохранить препроцессором элементы типа "Флеш-память" (можно подумать над "Сан.муфта")
#проверить словарь тегов по расстоянию jaro (сателлит и саттелит) если 0.8 то из 2х делай 1




'''
1)+
если аналитик не знает как классифицировать, 
или код не удовлетворяет формату счета из словаря счетов, 
ставь UNK

2)+
    Выделять для тегов только первые 3 слова в предложениее
preprocessor
исследовать три первых слова для тега 

3)
    Проверить теги на близость друг другу
Расстояние левинштейна, 
доп проверка, были ли использованы в классификации следующие строки

4)
удалить дубли в qqqq

5)+
вывести все в файл и сохранить файл

6)+
оптимизировать и причесать код


'''


#Доклад о проделанной работе за полгода:
'''

Доклад о проделанной работе за полгода:

    Был написан препроцессор для номенклатурных данных взамен старого
    (реализованы ранее отсутствующие функции сохранения биграмм и колокаций)
    
    Разобран реализации метода word2vec на базе НС Тани
    Переписал проект MIA preprocessing (предобработка данных для классификатора)
    Тестирование проекта MIA preprocessing, подбор гиперпарамератров для обучения НС
    
    Тестирование МИА, проверка отработки на входных данных, проверка классификации данных, 
    по средствам обращения к серверу.
    
    Теория по НС, курс по DL Козлова Новосибирского гос универа, книга по глубокому обучению
    
    Разметка данных по номенклатурным данным, выделение именованных сущностей (Создание словаря ТЕГов)
    
    Самостоятельно написал Word2Vec на TensorFlow

'''




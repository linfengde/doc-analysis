#-*- coding:utf-8 _*-
"""
@author:Fengde Lin
@file: utils.py
@time: 2019/03/28
"""

import re
import regex
import time
from pprint import pprint as p
import numpy as np
import jieba
import codecs
import pickle
import math
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def log(func):
    def wrapper(*args, **kwargs):
        now_time = str(time.strftime('%Y-%m-%d %X', time.localtime()))
        print('------------------------------------------------')
        print('%s %s called' % (now_time, func.__name__))
        print('Document:%s' % func.__doc__)
        print('%s returns:' % func.__name__)
        re = func(*args, **kwargs)
        p(re)
        return re
    return wrapper


def get_set_key(dic,threshold):
    '''选取频数大于等于Threshold的关键词构建一个集合，用于作为共现矩阵的首行和首列'''
    wf = {k: v for k, v in dic.items() if v >= threshold}
    set_key_list=[]
    for a in sorted(wf.items(), key=lambda item: item[1], reverse=True):
        set_key_list.append(a[0])
    print('过滤出现频次低于{}次的词，剩余关键词共有{}'.format(threshold,len(set_key_list)))
    return set_key_list


def format_data(data,set_key_list):
    '''格式化需要计算的数据，将原始数据格式转换成二维数组'''
    formated_data = []
    for ech in data:
        # ech_line = ech.split('/')
        ech_line = ech

        temp = []            # 筛选出format_data中属于关键词集合的词
        for e in ech_line:
            if e in set_key_list:
                temp.append(e)
        ech_line=temp

        ech_line = list(set(filter(lambda x: x != '', ech_line)))  # set去掉重复数据
        formated_data.append(ech_line)
    return formated_data


def count_matrix(matrix, formated_data):
    '''计算各个关键词共现次数'''
    keywordlist=matrix[0][1:]  #列出所有关键词
    appeardict={}  #每个关键词与 [出现在的行(formated_data)的list] 组成的dictionary
    for w in keywordlist:
        appearlist=[]
        i=0
        for each_line in formated_data:
            if w in each_line:
                appearlist.append(i)
            i +=1
        appeardict[w]=appearlist
    for row in range(1, len(matrix)):
        # 遍历矩阵第一行，跳过下标为0的元素
        for col in range(1, len(matrix)):
                # 遍历矩阵第一列，跳过下标为0的元素
                # 实际上就是为了跳过matrix中下标为[0][0]的元素，因为[0][0]为空，不为关键词
            if col >= row:
                #仅计算上半个矩阵
                if matrix[0][row] == matrix[col][0]:
                    # 如果取出的行关键词和取出的列关键词相同，则其对应的共现次数为0，即矩阵对角线为0
                    matrix[col][row] = str(0)
                else:
                    counter = len(set(appeardict[matrix[0][row]])&set(appeardict[matrix[col][0]]))

                    matrix[col][row] = str(counter)
            else:
                matrix[col][row]=matrix[row][col]
    return matrix


def build_matirx(set_key_list):
    '''建立矩阵，矩阵的高度和宽度为关键词集合的长度+1'''
    edge = len(set_key_list)+1
    # matrix = np.zeros((edge, edge), dtype=str)
    matrix = [['' for j in range(edge)] for i in range(edge)]
    return matrix


def init_matrix(set_key_list, matrix):
    '''初始化矩阵，将关键词集合赋值给第一列和第二列'''
    matrix[0][1:] = np.array(set_key_list)
    matrix = list(map(list, zip(*matrix)))
    matrix[0][1:] = np.array(set_key_list)
    return matrix


def doc_cut(doc,keywords_list=None):
    '''
    :param doc: str
    :return: ['word1','word2',...]
    '''
    import jieba
    if keywords_list:
        for word in keywords_list:
            jieba.add_word(word)

    seg_text = []
    for word in jieba.cut(doc, cut_all=False):
        if not word.strip():
            continue
        seg_text.append(word)
    return seg_text

# 计算PMI



# 计算LDA困惑度
def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    """calculate the perplexity of a lda-model"""
    # dictionary : {7822:'deferment', 1841:'circuitry',19202:'fabianism'...]
    print ('the info of this ldamodel: \n')
    print ('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0 # the num of words in the doc
        for word_id, num in doc:
            prob_word = 0.0 # the probablity of the word
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    print ("the perplexity of this ldamodel is : %s"%prep)
    return prep


def get_word_position(list_position_name,keywords_list):
    list_price_positoin_address = []
    for i in list_position_name:
        address_index = [x for x in range(len(list_position_name)) if list_position_name[x] == i]
        list_price_positoin_address.append([i, address_index])
    dict_address = dict(list_price_positoin_address)
    dict_address_keywords = { k:v for k,v in dict_address.items() if k in keywords_list}
    return dict_address_keywords


def get_word_group(word,position_dict,word_list,num):
    word_position = position_dict[word]
    word_group_all = []
    for position in word_position:
        # try:
        word_before = word_list[position-3]+word_list[position-2] + word_list[position-1]
        word_group = word_before[-num:] + word_list[position]
        # except:
        #     word_group = word_list[position]
        # word_group = ''.join(word_list[position-2:position])
        word_group_all.append(word_group)
    return word_group_all


def get_true_index(input_str,keywords_list):
    input_str = re.sub('\s', '', input_str) # 将空白字符替换，因为正则会算空白字符，分词不包含空白字符，导致索引可能对不上
    # input_str = input_str.replace('\n','')
    word_list = doc_cut(input_str)
    keywords_position = get_word_position(word_list,keywords_list)

    result_dict = {}     # {'word':}
    for word in keywords_list:
        word_all_index = re.findall(word,input_str)
        if word_list.count(word) == len(word_all_index):
            result_dict[word] = list(range(1,len(word_all_index)+1))
        else:
            # 分别比较正则和分词  关键字和前3个字组合（保证唯一性）后是否一样,        三个前提 1. 通过前3个字符组合保证不重复   2. 正则出来的关键字包含所有分词结果   3.该词在至少三个字以后出现
            re_list = regex.findall('...'+word, input_str[-3:]+input_str, overlapped=True) #考虑重叠和关键词在文章开头的地方
            jieba_list = get_word_group(word,keywords_position,word_list,3)
            true_index = []
            for word_group in jieba_list:
                re_list_index = re_list.index(word_group)
                true_index.append(re_list_index+1)
                re_list[re_list_index] = ''
            result_dict[word] = true_index

    return result_dict

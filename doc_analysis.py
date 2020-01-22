#-*- coding:utf-8 _*-
"""
@author:Lengde Lin
@file: main.py
@time: 2019/12/25
@function: v1.2:   测试版
******* 功能点 ********
1.近义词词频统计              word_count
2.共现词发现                  concurrence_word_count
3.比较2篇文章新增词           newword_count
4.比较2篇文章相同词变化趋势   sameword_count
5.新词、短语发现功能          newword_found
6.差异化比较                  difference_count
7.查找关键词所在句子          find_sentences
8.生成文章关键词              enerate_keywords

******* 词性查找表 ********
https://www.cnblogs.com/adienhsuan/p/5674033.html
"""

# import sys
# sys.path.append('./lib')
import sys
import os
sys.path.append(os.path.dirname(__file__))

from simword_count import SimWordCount
import jieba.analyse
import jieba.posseg as pseg
import pandas as pd
import jieba
import itertools
import re
import copy
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer


# 词特征
# TODO: 引入PMI互信息， 词跨度，位置信息（基于文档位置）

# 统计词频
def word_count(input_str,count_only_singleword=False,calculate_tfidf_pseg=False,keywords_list=None):
    '''
    :param input_txt:输入文档
    :param count_only_singleword:  TRUE： 统计单个词     False: 近义词群
    :param calculate_tfidf_pseg:  返回结果 是否包含词性，TFIDF值
    :param keywords_list: 关键词列表
    :return:   1.calculate_tfidf_pseg=TRUE    ['keys', 'value','tfidf', 'tag','length', 'weights']    # weight= value* tfidf
                2.calculate_tfidf_pseg=False    ['keys', 'value']
    '''

    word_frequency_count = SimWordCount(input_str,keywords_list)
    keys, values = word_frequency_count.count(input_str,'False',30,1,count_only_singleword=count_only_singleword)
    # 返回全部词（topn无效）  不使用黑名单功能（不统计近义词）
    data = {'keys':keys,'value':values}
    result = pd.DataFrame(data)

    if calculate_tfidf_pseg:
        # 词性标注
        result_dict = {key: value for key, value in zip(result['keys'], result['value'])}
        words_pseg = pseg.cut(input_str)
        words_pseg_tuple = [(word, pseg) for word, pseg in words_pseg]
        # words_pseg_tuple = [(word, pseg) for word, pseg in words_pseg if pseg in ['x','n','nr','ns','nt','nz','nrfg']]   # 词性过滤
        words_tag = {}
        for key in keys:
            for (word, tag) in set(words_pseg_tuple):
                if key != word:
                    continue
                else:
                    words_tag[key] = tag
            if key not in words_tag.keys():  # 无法被jieba分出来
                words_tag[key] = 'unknown'
        tags = list(words_tag.values())

        # TFIDF值计算
        words_weight = jieba.analyse.extract_tags(input_str, topK=100000000, withWeight=True)
        words_weight_tuple = [(word, weight) for word, weight in words_weight if word in result_dict.keys()]
        words_tfidf = {}
        for key in keys:
            for (word, weight) in words_weight_tuple:
                if key != word:
                    continue
                else:
                    words_tfidf[key] = weight
            if key not in words_tfidf.keys():  # 无法被jieba分出来
                words_tfidf[key] = 0
        tfidfs = list(words_tfidf.values())

        # 词的长度
        keys_size = [len(_) for _ in keys]

        # 词的跨度因子
        words_span = word_frequency_count.count_span()
        words_span_tuple = {word:span for word,span in words_span.items() if word in result_dict.keys()}
        spans = list(words_span_tuple.values())

        # 计算权重
        # weights = [tfidf*length for length, tfidf in zip(keys_size, tfidfs)]   # 权重= 字符长度* tfidf
        weights = [0.05*freq+0.4*tfidf+0.4*length+0.15*span for freq, length, tfidf, span in zip(values, keys_size, tfidfs,spans)]

        data = {'keys': keys, 'value': values,  'tfidf': tfidfs, 'tag': tags,'length':keys_size,'span':spans, 'weights': weights}
        result = pd.DataFrame(data, columns=['keys', 'value',  'tfidf', 'tag','length','span', 'weights'])
        result.sort_values(by='weights', ascending=False, inplace=True)
        # result.query('weights>0.1', inplace=True)
        return result

    return result


# 比较2篇文章,统计第2篇文章中新出来的词
def newword_count(compare_txt1,compare_txt2,output_csv):
    '''
    :param compare_txt1: 分析文档
    :param compare_txt2: 被比较文档
    :param output_csv: 输出CSV路径
    :return: dataframe       word value      关键词 出现次数
    '''
    result1 = word_count(compare_txt1,count_only_singleword=True)
    result2 = word_count(compare_txt2,count_only_singleword=True)

    result1_dict = { key:value for key,value in zip(result1['keys'],result1['value'])}
    result2_dict = { key:value for key,value in zip(result2['keys'],result2['value'])}

    # 统计新出现词的出现频率
    new_key = [key for key in result2_dict.keys() if key not in result1_dict.keys()]
    newword_count = {key:value for key,value in result2_dict.items() if key in new_key}
    newword_count = sorted(newword_count.items(), key=lambda x: x[1], reverse=True)
    data = {'word':[group[0] for group in newword_count],'freq':[group[1] for group in newword_count]}
    newword_result = pd.DataFrame(data)
    newword_result.to_csv(output_csv, index=False, encoding='gb18030')

    return newword_result


# 比较2篇文章中相同词变化趋势
def sameword_count(compare_txt1,compare_txt2,output_csv):
    '''
    :param compare_txt1: 分析文档
    :param compare_txt2: 被比较文档
    :param output_csv: 输出CSV路径
    :return: dataframe       word value1 value2 diff         关键词 出现次数1 出现次数2    差值
    '''
    result1 = word_count(compare_txt1,output_csv='',write_csv=False,count_only_singleword=True)
    result2 = word_count(compare_txt2,output_csv='',write_csv=False,count_only_singleword=True)

    result1_dict = { key:value for key,value in zip(result1['keys'],result1['value'])}
    result2_dict = { key:value for key,value in zip(result2['keys'],result2['value'])}

    # 统计以前出现过词的变化频率
    same_key = [key for key in result2_dict.keys() if key in result1_dict.keys()]
    values_group = [(value1,value2,value2-value1,key1) for key1,value1,value2 in zip(result2_dict.keys(),result1_dict.values(),result2_dict.values()) if key1 in same_key and value2!=value1]
    value1s = [group[0] for group in values_group]
    value2s = [group[1] for group in values_group]
    values_diff = [group[2] for group in values_group]
    keys = [group[3] for group in values_group]
    data2 = {'word':keys,'value1':value1s,'value2':value2s,'diff':values_diff}
    sameword_result = pd.DataFrame(data2)
    sameword_result.to_csv(output_csv, index=False, encoding='gb18030')

    return sameword_result


# 新词发现功能，需要人工辅助判断
def newword_found(input_str,mode='all',filter=True):
    '''
    :param input_txt:分析文档 TXT格式
    :param mode: 新词发现模式  all: 计算DOA,DOF，TFIDF,freq   part：只考虑TFIDF，freq
    :param filter:
    :return:
    '''
    from termrecognition import termsRecognition

    # 分句多行list
    input_list = set(re.split('[?!…。？！]', input_str))  # 去重

    generator = termsRecognition(content=input_list, is_jieba=False, topK=30, mode=[1])  # 文字版

    if mode=='all':                  # 全部发现------统计freq,doa,dof,idf计算指标
        result_dict = generator.generate_word()
        result_dataframe = generator.get_result()
    elif mode == 'part':             # 部分发现------统计freq,idf计算指标
        result_dataframe = generator.part_found()
    else:
        raise ValueError('use all or part for mode parameter')

    if filter:
        filter_fn = os.sep.join([os.path.dirname(__file__), 'lib','forbidden_all.txt'])
        stopwords = [line.strip() for line in open(filter_fn,'r', encoding='utf-8').readlines()]# 采用停用词
        newwords=[];dofs=[];freqs=[];scores=[];idfs=[]
        for index,row in result_dataframe.iterrows():
            word_group = row['key'].split(' ')
            word1 = word_group[0]
            word2 = word_group[1]
            new_word = word1 + word2
            if len(word1)<=1 or len(word2)<=1 or word1<= '\u4e00' or word1>= '\u9fff' or word2<= '\u4e00' or word2>= '\u9fff' or word1.strip() in stopwords or word2.strip() in stopwords or new_word.strip() in stopwords:
                continue
            else:
                newwords.append(word_group[0]+word_group[1])
                dofs.append(row['dof'])
                freqs.append(row['freq'])
                scores.append(row['dof']*row['freq']*row['idf'])    # 权重 = 出现频率*IDF值*DOF值
                idfs.append(row['idf'])

        newword_data={'keyword':newwords,'dof':dofs,'freqs':freqs,'score':scores,'idf':idfs}
        result = pd.DataFrame(newword_data,columns=['keyword','score','freqs','dof','idf'])
        result.sort_values(by='score',ascending=False,inplace=True)
        result.query('score>0',inplace=True)

    return result


# 统计文章中共现词
def concurrence_word_count(input_txt,output_csv,threshold =4, write_csv=False,write_node_edge=False,keywords=None):

    fie = [line.strip() for line in open(input_txt, 'r', encoding='utf-8').readlines() if len(line) > 1]    # 按特殊标点符号，分割句子
    article =''.join(fie)
    data = re.split('[?!…。？！*]', article)

    data_cut_list = []

    for item in data:
        item_list = doc_cut(item,keywords)
        item_fixed_list = [word for word in item_list if len(word)>1 and word>= '\u4e00' and word<= '\u9fff']   # 去除单字符，且不为中文
        data_cut_list.append(item_fixed_list)

    data = word_count(input_txt, output_csv='', write_csv=False,count_only_singleword=True)
    keywords_dict = {k:v for k,v in zip(data['keys'],data['value'])}

    set_key_list = get_set_key(keywords_dict,threshold)
    formated_data = format_data(data_cut_list,set_key_list)
    matrix = build_matirx(set_key_list)
    matrix = init_matrix(set_key_list, matrix)
    result_matrix = count_matrix(matrix, formated_data)

    df_result = pd.DataFrame(result_matrix)
    if write_csv:
        df_result.to_csv(output_csv,index=False,encoding='gb18030')

    # 改写成边节点文件
    if write_node_edge:
        row_1 = list(df_result.iloc[0,:])
        column_1 = list(df_result.iloc[:,0])
        keywords_combination = list(itertools.combinations(set_key_list,2))
        sources = []
        targets = []
        weights = []
        for combine in keywords_combination:
            row_index = row_1.index(combine[0])
            column_index = column_1.index(combine[1])
            combine_value = df_result.iloc[row_index,column_index]
            sources.append(combine[0])
            targets.append(combine[1])
            weights.append(int(combine_value))

        node_edge_data ={'Source':sources,'Target':targets,'Weight':weights}
        node_edge_fn = pd.DataFrame(node_edge_data)
        node_edge_fn.query('Weight>0',inplace=True)    # 过滤共现次数>1
        if write_csv:
            node_edge_fn.to_csv('./result/node_edge.csv',index=False,encoding='gb18030')
        df_result = node_edge_fn

    return df_result


# 统计文章中差异性
def difference_count(compare_txt1,compare_txt2,output_csv,output_csv2,write_csv=False,count_concurrence_diff=False):
    '''
    :param compare_txt1:被比较文章
    :param compare_txt2:主要分析文章
    :param output_csv: 输出路径
    :param write_csv: 是否写入CSV文件
    :return:
    '''
    newword_result1 = newword_found(compare_txt1, '', mode='all', filter=True,write_csv=False)
    newword_result2 = newword_found(compare_txt2, '', mode='all', filter=True,write_csv=False)
    newword_result1_dict = { key:value for key,value in zip(newword_result1['keyword'],newword_result1['freqs'])}
    newword_result2_dict = { key:value for key,value in zip(newword_result2['keyword'],newword_result2['freqs'])}
    # TODO: 把新词发现结果加入jieba会影响最终分词效果， 根据需求权衡
    keywords_list1 = [word for word in list(newword_result1['keyword'])]     # 将新词加入词频统计中
    keywords_list2 = [word for word in list(newword_result2['keyword'])]     # 将新词加入词频统计中

    word_count_result1 = word_count(compare_txt1, '', write_csv=False, count_only_singleword=True,calculate_tfidf_pseg=True,keywords_list=None)      # calculate_tfidf_pseg 是否计算TFIDF和词性，过滤
    word_count_result2 = word_count(compare_txt2, '', write_csv=False, count_only_singleword=True,calculate_tfidf_pseg=True,keywords_list=None)
    word_count_result1_dict = { key:value for key,value in zip(word_count_result1['keys'],word_count_result1['value'])}
    word_count_result2_dict = { key:value for key,value in zip(word_count_result2['keys'],word_count_result2['value'])}

    # 将词频+新词发现组合，去重后，组成字典
    result1_dict = dict(newword_result1_dict,**word_count_result1_dict)      # 如果键值一样，以词频统计为准
    result2_dict = dict(newword_result2_dict,**word_count_result2_dict)

    #TODO： 词性过滤 可注释
    # 统计相同词的变化频率
    same_key = [key for key in result2_dict.keys() if key in result1_dict.keys()]
    same_key_weights = [weight for key, weight in zip(word_count_result2['keys'],word_count_result2['weights']) if key in same_key]   # 按分析文章中的权重算
    same_key_tags = [tag for key, tag in zip(word_count_result2['keys'], word_count_result2['tag']) if key in same_key]
    # values_group = [(value1,value2,value2-value1,key1) for key1,value1,value2 in zip(result2_dict.keys(),result1_dict.values(),result2_dict.values()) if key1 in same_key]
    values_group = [(value1,value2,value2-value1,key1) for key1,value1,value2 in zip(word_count_result2['keys'],word_count_result1['value'],word_count_result2['value']) if key1 in same_key]   # 此处忽略了新词发现后的 相同KEY的情况，选择取消注释上一句，并不考虑WEIGHT和TAG
    value1s = [group[0] for group in values_group]
    value2s = [group[1] for group in values_group]
    values_diff = [group[2] for group in values_group]
    keys = [group[3] for group in values_group]
    data = {'word':keys,'value1':value1s,'value2':value2s,'diff':values_diff,'tag':same_key_tags,'weight':same_key_weights}
    sameword_result = pd.DataFrame(data,columns=['word','diff','value2','value1','tag','weight'])
    sameword_result.sort_values(by='weight', ascending=False, inplace=True)

    # 统计新发现词的出现频率
    notsame_keys = [key for key in result2_dict.keys() if key not in result1_dict.keys()]    # 统计不同词
    notsame_keys_size = [len(_) for _ in notsame_keys]
    notsame_values = [value for key,value in result2_dict.items() if key in notsame_keys]
    weights = [size * freq for freq, size in zip(notsame_keys_size, notsame_values)]    # 权重= 长度* tfidf
    data2 = {'word':notsame_keys,'freq':notsame_values,'length':notsame_keys_size,'weight':weights}
    notsameword_result = pd.DataFrame(data2,columns=['word','freq','length','weight'])
    notsameword_result.sort_values(by='weight', ascending=False, inplace=True)
    notsameword_result.query('freq>1',inplace=True)

    # 计算权重高的词，共现词对的差异化
    if count_concurrence_diff:

        concurrence_group2 = concurrence_word_count(compare_txt2, '', threshold=1, write_csv=False,write_node_edge=True,keywords=None)
        concurrence_group1 = concurrence_word_count(compare_txt1, '', threshold=1, write_csv=False,write_node_edge=True,keywords=None)
        concurrence_group2_dict = {(source,target):weight for source, target,weight in zip(concurrence_group2['Source'],concurrence_group2['Target'],concurrence_group2['Weight'])}
        concurrence_group1_dict = {(source,target):weight for source, target,weight in zip(concurrence_group1['Source'],concurrence_group1['Target'],concurrence_group1['Weight'])}

        concurrence_diff_dict = {}
        top_samewords = sameword_result['word'][0:40]        # 选取相同关键词权重高的TOP词
        diff_value = [(k2, v2) for k1, k2, v1, v2 in zip(concurrence_group1_dict.keys(), concurrence_group2_dict.keys(),
                                                         concurrence_group1_dict.values(),
                                                         concurrence_group2_dict.values()) if k2 != k1]
        # 权重词中词共现对
        for word in top_samewords:
            word_group=[]
            for item in diff_value:
                if word == item[0][0]:
                    word_group.append(item)
                elif word == item[0][1]:
                    word_group.append(((item[0][1],item[0][0]),item[1]))
                else:
                    continue
            concurrence_diff_dict[word] = copy.deepcopy(word_group)

        a = [v for v in list(concurrence_diff_dict.values()) if v]      # 去掉没有共现词差的集合
        b = [_ for item in a for _ in item]
        group = [item[0] for item in b]
        word_freqs = [item[1] for item in b]      # 出现频率
        sources = [item[0] for item in group]
        targets = [item[1] for item in group]   # 共现词2
        word_weight_dict = {key: (value,tag) for key, value,tag in zip(word_count_result2['keys'], word_count_result2['weights'], word_count_result2['tag'])}
        target_weights = [v[0] for word in targets for k,v in word_weight_dict.items() if word == k]   # 找出共现词2的权重
        target_tags = [v[1] for word in targets for k,v in word_weight_dict.items() if word == k]   # 找出共现词2的词性

        # 不需要词性过滤，请注释下面2句
        data_group = [(s,t,f,tw,tt) for s,t,f,tw,tt in zip(sources,targets,word_freqs,target_weights,target_tags) if tt in ['j','l','m','x','n','nr','ns','nt','nz','nrfg']]
        sources = [item[0] for item in data_group];targets = [item[1] for item in data_group];word_freqs = [item[2] for item in data_group];target_weights = [item[3] for item in data_group];target_tags = [item[4] for item in data_group];

        data = {'source': sources,'target': targets,'freq': word_freqs,'target_weights':target_weights,'target_tags':target_tags}
        diff_wordgroup = pd.DataFrame(data,columns=['source','target','freq','target_weights','target_tags'])
        diff_wordgroup.sort_values(by='target_weights', ascending=False, inplace=True)
        diff_wordgroup.query('freq>0',inplace=True)
        diff_wordgroup.to_csv('./result/diff_group_count.csv', index=False, encoding='gb18030')

    if write_csv:
        sameword_result.to_csv(output_csv, index=False, encoding='gb18030')
        notsameword_result.to_csv(output_csv2, index=False, encoding='gb18030')

    return sameword_result


# 返回包含指定关键词所在句
def find_sentences(keywords_list, input_str, topn):
    from flashtext.keyword import KeywordProcessor
    from collections import Counter
    '''
    :param keywords_list:  list     ['word1',..'wordn']
    :param doc_txt: txt文档
    :param topn: 显示重要程度TOPN的句子
    :return:keywords_sentences: dict  {'word1':['sen1','sen2',...],...}
    '''
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(keywords_list)

    # 按标点符号切分句子
    doc_cut_list = set(re.split('[?!…。？！]', input_str))  # 去重

    # 查找关键词所在句子
    sentences_wordcount={}
    for sentence in doc_cut_list:
        keywords_found = keyword_processor.extract_keywords(sentence)
        if len(keywords_found)!=0:
            keywords_count = Counter(keywords_found)
            sentences_wordcount[sentence] = keywords_count

    keywords_sentences = {}
    for word in keywords_list:
        keywords_sentences[word] = [k for k, v in sentences_wordcount.items() if word in v.keys()][0:topn]

    return keywords_sentences


# 查找包含新增概念权重高的句子
def find_sentences_weight(word_weight_dict, input_str,topn):
    from flashtext.keyword import KeywordProcessor
    from collections import Counter
    '''
    :param keywords_list:  list     ['word1',..'wordn']
    :param top_keywords:  计算TOP权重的关键词     
    :param input_str: str文档
    :param topn: 显示重要程度TOPN的句子
    :return:keywords_sentences: dict  {'word1':['sen1','sen2',...],...}
    '''
    # 读取词汇权重文件

    keywords_list = list(word_weight_dict.keys())

    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(list(keywords_list))

    input_str = re.sub('\s', '', input_str)
    doc_cut_list = set(re.split('[?!…。；？！]',input_str))   # 去重

    # 计算句子重要程度
    sentences_wordcount={}       # [('句子',(value,['word1',...'wordn']))]
    for sentence in doc_cut_list:
        keywords_found = keyword_processor.extract_keywords(sentence)
        if len(keywords_found)!=0:
            keywords_in_sentence = []
            sentence_value = 0
            keywords_count = Counter(keywords_found)
            for k,v in keywords_count.items():
                sentence_value = sentence_value+v*word_weight_dict[k]                   # 权重 = 求和(句子中出现的关键词权重)
                keywords_in_sentence.append(k)
            sentence_value = len(keywords_in_sentence)*sentence_value/len(sentence)         #  权重 = 关键词个数 * 关键词权重 / 句子长度，  可根据需要注释
            sentences_wordcount[sentence] = (sentence_value,keywords_in_sentence)

    sentences_wordcount_sort = sorted(sentences_wordcount.items(),key=lambda x:x[1][0], reverse=True)    # 句子，权重     [('句子',(value,['word1',...'wordn']))]
    topn_sentences = [s[0] for s in sentences_wordcount_sort[0:topn]]

    return topn_sentences


# 提取包含重要词的句子
def find_sentences_rule(keywords_list,doc_txt):
    from flashtext.keyword import KeywordProcessor

    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(keywords_list)

    # 按标点符号切分句子
    fie = [line.strip() for line in open(doc_txt, 'r', encoding='utf-8').readlines() if len(line) > 1]    # 按特殊标点符号，分割句子
    article = ''
    for i in fie:
        article += i.strip()+'*'
    doc_cut_list = re.split('[?!…。？！*]',article)       # 对不同行，没有标点的句子通过‘*

    sentences_important = []
    for sentence in doc_cut_list:
        keywords_found = keyword_processor.extract_keywords(sentence)
        if len(keywords_found) != 0:
            sentences_important.append(sentence)
    print(sentences_important)
    return sentences_important


# 生成文章的关键词
def generate_keywords(input_txt, output_csv,write_csv=True):

    # 词频统计
    input_str = [line.strip() for line in open(input_txt, 'r', encoding='utf-8').readlines() if len(line) > 1]    # 按特殊标点符号，分割句子
    input_str = ''.join(input_str)
    word_count_result = word_count(input_str, count_only_singleword=True)
    word_count_result_dict = { key:value for key,value in zip(word_count_result['keys'],word_count_result['value'])}

    # 新词发现
    newword_result = newword_found(input_str, filter=True)
    newword_result_dict = { key:value for key,value in zip(newword_result['keyword'],newword_result['freqs'])}

    # 将词频+新词发现组合，去重后，组成字典
    result_dict = dict(newword_result_dict,**word_count_result_dict)      # 如果键值一样，以词频统计为准

    # 统计新发现词的出现频率
    keys = [word for word in result_dict.keys()]
    keys_size = [len(_) for _ in result_dict.keys()]
    keys_freq = [value for value in result_dict.values()]

    # 标注词性,TFIDF计算
    for word in result_dict.keys():
        jieba.add_word(word)

    fie = [line.strip() for line in open(input_txt, 'r', encoding='utf-8').readlines() if len(line) > 1]    # 按特殊标点符号，分割句子
    article =''.join(fie)

    # 词性标注
    words_pseg = pseg.cut(article)
    words_pseg_tuple = [(word, pseg) for word,pseg in words_pseg if word in result_dict.keys()]  # 需要去重
    words_tag = {}
    for key in keys:
        for (word,tag) in set(words_pseg_tuple):
            if key != word:
                continue
            else:
                words_tag[key] = tag
        if key not in words_tag.keys():  # 无法被jieba分出来
            words_tag[key] = 'unknown'
    tags = list(words_tag.values())


    # TFIDF值计算
    words_weight = jieba.analyse.extract_tags(article, topK=100000000, withWeight=True)
    words_weight_tuple = [(word, weight) for word,weight in words_weight if word in result_dict.keys()]
    words_tfidf = {}
    for key in keys:
        for (word,weight) in words_weight_tuple:
            if key != word:
                continue
            else:
                words_tfidf[key] = weight
        if key not in words_tfidf.keys():  # 无法被jieba分出来
            words_tfidf[key] = 0
    tfidfs = list(words_tfidf.values())

    weights = [size * tfidf for freq, size,tfidf in zip(keys_freq, keys_size,tfidfs)]    # 权重= 长度* tfidf

    # 如果不需要过滤词性，可以将下面2句注释掉
    group = [(key,freq,size,tfidf,tag,weight)for key, freq,size,tfidf,tag,weight in zip(keys,keys_freq, keys_size,tfidfs,tags,weights) if tag in ['x','n','nr','ns','nt','nz','nrfg']]
    keys = [item[0] for item in group];keys_freq =[item[1] for item in group];keys_size =[item[2] for item in group];tfidfs =[item[3] for item in group];tags =[item[4] for item in group];weights =[item[5] for item in group];

    data2 = {'word':keys,'freq':keys_freq,'length':keys_size,'tfidf':tfidfs,'tag':tags,'weights':weights}
    keywords_result = pd.DataFrame(data2,columns=['word','freq','length','tfidf','tag','weights'])
    keywords_result.sort_values(by='weights', ascending=False, inplace=True)
    keywords_result.query('freq>1',inplace=True)   # 过滤机制

    if write_csv:
        keywords_result.to_csv(output_csv, index=False, encoding='gb18030')

    # 输出主题,由权重高的5个关键词组合
    theme=''
    for word in keywords_result['word'][0:5]:
         theme = theme+word+'#'
    #theme = keywords_result['word'][0:5]
    return keywords_result,theme


def tfidf_cosine(s1, s2):
    vectorizer = TfidfVectorizer()
    train_modle = vectorizer.fit([s1, s2])
    print(train_modle.vocabulary_)
    X = train_modle.transform([s1, s2])  # 得到s1，s2用TF-IDF方式表示的向量
    print(X.toarray())
#    print(cosine_similarity(X[0], X[1]))




# TODO： 通过LDA提取主题词
# LDA文档主题分类
# def lda_topic():

# TODO: 通过图算法提取主题词

if __name__ == '__main__':
    # input_txt = './data/2018future.txt'     # 主要分析文件，新增词频，相同词频基于这个文件
    # input_txt2 = './data/2017future.txt'      # 被比较文章
    input_txt = './data/test.txt'     # 主要分析文件，新增词频，相同词频基于这个文件

    # input_txt = './data/airline2.txt'
    # input_txt2 = './data/airline1.txt'  # 被比较文章
    # input_txt = './data/2015deriatives.txt'          #  span有BUG
    # input_txt2 = './data/2014deriatives.txt'  # 被比较文章

    # ************ 近似词频统计 ******************
    # output_word_count_csv = './result/word_count.csv'
    # word_count_result = word_count(input_txt,output_word_count_csv,write_csv=True,count_only_singleword=True,calculate_tfidf_pseg=True)

    # ************ 共现词频统计 ******************
    # output_concurrence_count_csv = './result/concurrence_count.csv'
    # concurrence_word_count_result = concurrence_word_count(input_txt,output_concurrence_count_csv,threshold =1,write_csv=False,write_node_edge=False)

    # ************ 新增词频统计 ******************
    # output_newword_count_csv = './result/newword_count.csv'
    # newword_count_result = newword_count(input_txt2,input_txt,output_newword_count_csv)

    # ************ 相同词频变化趋势统计 ******************
    # output_sameword_csv = './result/sameword_count.csv'
    # sameword_count_result = sameword_count(input_txt2,input_txt,output_sameword_csv)

    # ************ 新词发现 ******************
    # output_termfound_csv = './result/term_result_all.csv'
    # newword_found_result = newword_found(input_txt,output_termfound_csv,mode='all',filter=True,write_csv=True)

    # ************ 差异化比较 ******************
    # output_difference_count_csv = './result/compare_count.csv'
    # output_notsame_count_csv = './result/notsame_count.csv'
    # difference_count_result = difference_count(input_txt2,input_txt,output_difference_count_csv,output_notsame_count_csv,write_csv=True,count_concurrence_diff=True)

    # ************ 查找关键词所在关键句 ******************
    # keywords = ['开放型经济','大豆','国际期货大会','贸易保护主义','有色金属产业']
    # output_csv = './result/keywords_sentences.csv'
    # find_sentences(keywords,input_txt,output_csv,topn=2,write_csv=True)

    # ************ 排序新增概念句子 ******************
    # word_weight_csv = './result/notsame_count.csv'              # 根据difference_count得出
    # output_sentence_sort_csv ='./result/sentence_sort.csv'
    # topn_sentences = find_sentences_weight(word_weight_csv, input_txt,output_sentence_sort_csv,top_keywords=20,topn=2,write_csv=True)

    # ************ 根据规则筛选句子（生成目录） ******************
    # keywords_path = './lib/keywords.txt'
    # keywords_list = [line.strip() for line in open(keywords_path,'r',encoding='utf-8').readlines()]
    # sentences_rule = find_sentences_rule(keywords_list, input_txt2)

    # ************ 生成文章关键词 ******************
    # output_keywords_csv ='./result/keywords_weight.csv'

    output_keywords_csv = None
    keywords_result,theme = generate_keywords(input_txt,output_keywords_csv,write_csv=False)
    print(keywords_result)
    print(theme)
#    test_keywords_result = keywords_result['word'][:]

#    print(tfidf_cosine("二代稳性在海工平台及海工船的适用范围及稳性计算实现",
#                      "通过开展国内外海工行业项目生产组织模式收集、调研、分析、研究，为招商局重工生产组织模式变革、精益化生产提供理论分析基础。"))
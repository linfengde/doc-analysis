#-*- coding:utf-8 _*-
"""
@author:Yuefeng Lin
@file: word_count.py
@time: 2019/02/21
@v1.4  稳定版    专用于话题演变分析版本
@notice: 采用flashtext检索自定义近似词， 由于flashtext原理，不能保证完全切分出自定义词
"""
import os
import jieba
from flashtext.keyword import KeywordProcessor
from collections import Counter


class SimWordCount(object):

    def __init__(self,input,keywords_list=None):


        # 初始化词频统计类
        self.keyword_processor = KeywordProcessor()

        # 加载停用词词典
        stopwords = os.sep.join([os.path.dirname(__file__), 'lib', 'stopwords_all.txt'])
        if not os.path.exists(stopwords): raise FileNotFoundError('stopwords file is not found!')
        self.stopwords_list =[line.strip().decode('utf-8') for line in open(stopwords,'rb').readlines()]

        # 加入自定义词,辅助分词
        if keywords_list:
            for word in keywords_list:
                jieba.add_word(word)

        # 加载文档并分词
        self.seg_text = self._doc_cut(input)         # 分词

        # 加载近义词词典
        cilin = os.sep.join([os.path.dirname(__file__), 'lib', 'new_cilin.txt'])
        if not os.path.exists(cilin): raise FileNotFoundError('近义词词典new_cilin.txt is not found!')
        self.file = cilin
        self.word_code = {}
        self.code_word = {}
        self.vocab = set()
        self._read_cilin()

        # 加载预定义的近似词词典
        if os.path.exists(os.sep.join([os.path.dirname(__file__), 'lib','simword.txt'])):
            self.simword_group = [line.strip().decode('utf-8-sig') for line in open(os.sep.join([os.path.dirname(__file__),'lib','simword.txt']),'rb').readlines()]
            self.simword_group = [group.split() for group in self.simword_group]
        else:
            raise FileNotFoundError('可在当前文件夹下增加过滤词simword.txt,近义词中的词群将被列为统计结果')

        # 加载过滤词表
        if os.path.exists(os.sep.join([os.path.dirname(__file__), 'lib', 'black_list.txt'])):
            self.blacklist = [line.strip().decode('utf-8') for line in open(os.sep.join([os.path.dirname(__file__),'lib', 'black_list.txt']),'rb').readlines()]
        else:
            raise FileNotFoundError('可在当前文件夹下增加黑名单black_list.txt,黑名单中的词将不被列为统计结果')

        # 加载分词词典
        dict_path = os.sep.join([os.path.dirname(__file__), 'lib',  'new_dict.txt'])
        try:
            jieba.load_userdict(dict_path)
        except FileNotFoundError:
            print('WARNING: 为了更好的分词效果，建议在当前目录放置new_dict.txt自定义词典，辅助jieba分词')
        finally:
            pass

    def count(self, input, use_filter_words=True, topn=10, output_limit =1, count_only_singleword=True):
        '''
        :param : input:输入文本        type: str
        :param : thres:近义词阈值 0-1  type: float             词编码相同比例阈值, thres%相同，则认为是近义词
        :param : use_filter_words:是否使用过滤词表             True / False    默认为True
        :return: textgroup_count_sort 返回近义词组统计信息   type: list         [(words_group,count),...]
        '''
        # 输入参数类型检查
        if topn<1: raise ValueError('topn 必须大于等于1')
        if output_limit<1: raise ValueError('topn 必须大于等于1')

        if use_filter_words == 'True':
            use_filter_words = True
        elif use_filter_words == 'False':
            use_filter_words = False
        else:
            raise ValueError('use_filter_words必须是True或False')

        words = set(self.seg_text)                   # 原文中的全部词

        # 计算每个词的位置因子

        if count_only_singleword:      # 只统计单个词频
            words_from_dict = set()
        else:                          # 统计单个词，包括近义词群
            words_from_dict = set(self.word_code.keys())                                  # 近义词词典中包含的词（非重复）

        words_in_simgroup = set([word for elem in self.simword_group for word in elem])  # 自定义词表中包含的词（非重复）
        # words_notin_group = [[word] for word in list(words - words_from_dict-words_in_simgroup) if len(word) >output_limit and word[0]!='.']    # 不在1.近义词词典 2.自定义词群中的词（非重复）
        words_notin_group = [[word] for word in list(words - words_from_dict-words_in_simgroup) if len(word) >output_limit and word>= '\u4e00' and word<= '\u9fff']    # 不在1.近义词词典 2.自定义词群中的词（非重复） 去除数字，字符，单字

        # 将自定义词加入词典
        dict_simgroup_result = {}
        if self.simword_group:
            self.keyword_processor.add_keywords_from_list(list(words_in_simgroup))
            keywords_found = self.keyword_processor.extract_keywords(input)
            keywords_count = Counter(keywords_found)
            # 统计自定义词群频率
            for l in self.simword_group:
                word_count = 0
                for w in l:
                    word_count += keywords_count[w]
                if word_count != 0:
                    s = ','.join(l)
                    dict_simgroup_result[s] = word_count

        # 采用黑名单机制对返回结果词频进行过滤
        if use_filter_words :
            words_notin_filter = [word[0] for word in words_notin_group]
            words_notin_group = list(set(words_notin_filter)-set(self.blacklist))
            words_notin_group = [[word] for word in words_notin_group]
            words_from_dict = set(self.word_code.keys())- set(self.blacklist)

            self.simword_group = [list(set(group)-set(self.blacklist)) for group in self.simword_group]  # 如果自定义词表

        text_word_dict = {}
        for word in words:
            if word in words_from_dict:
                if len(word) > output_limit:
                    text_word_dict[word] = self.word_code[word]

        words_group = self._get_words_group(text_word_dict)       # 近义词词群
        words_group = words_notin_group + words_group             # 需要统计词频的全部词(普通词，近义词词群),自定义近义词由flashtext自己统计

        dict_result = {}
        count = Counter(self.seg_text)
        for l in words_group:
            word_count = 0
            for w in l:
                word_count += count[w]
            s = ','.join(l)
            dict_result[s] = word_count

        if self.simword_group:
            all_dict_result = dict(dict_result,**dict_simgroup_result)
        else:
            all_dict_result = dict_result
        textgroup_count_sort = sorted(all_dict_result.items(), key=lambda x: x[1], reverse=True)  # 按照出现频率排序的词群
        # keys = [group[0] for group in textgroup_count_sort[0:topn]]
        keys = [group[0] for group in textgroup_count_sort]
        values = [group[1] for group in textgroup_count_sort]
        # json_result = [{'key': group[0], 'value': group[1]} for group in textgroup_count_sort[0:topn]]
        return keys,values

    def count_span(self):
        # 计算每个词在文章中的跨度
        doc_cut = self.seg_text
        words = set(doc_cut)                   # 原文中的全部词

        span_word = {}
        sum_span = len(doc_cut)        # 文章中词的总个数
        for word in words:
            first = doc_cut.index(word)         # 在文章中首次出现的位置
            doc_cut.reverse()
            last = sum_span - doc_cut.index(word)          # 在文章中最后出现的位置
            doc_cut.reverse()
            span_word[word] = (last-first+1)/sum_span
        return span_word

    def _read_cilin(self):
        with open(self.file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                res = line.split()
                code = res[0]                 # 词义编码
                words = res[1:]               # 同组的多个词
                self.vocab.update(words)      # 一组词更新到词汇表中
                self.code_word[code] = words  # 字典，目前键是词义编码，值是一组单词。

                for w in words:
                    if w in self.word_code.keys():      # 最终目的：键是单词本身，值是词义编码。
                        self.word_code[w].append(code)   # 如果单词已经在，就把当前编码增加到字典中
                    else:
                        self.word_code[w] = [code]       # 反之，则在字典中添加该项。

    @staticmethod
    def _get_words_group(word_label_dict):
        words_group = []
        values = [t for t in set(tuple(_) for _ in word_label_dict.values())]  # 去重
        for value in values:
            words = [k for k,v in word_label_dict.items() if tuple(v)==value]   # 编码完全一样
            words_group.append(words)
        return words_group

    @staticmethod
    def _read_words_from_file(fn):
        f = open(fn,encoding='utf-8').readlines()
        words = [s.strip() for s in f if s.strip()]
        return words

    def _doc_cut(self, doc):
        seg_text = []
        for word in jieba.cut(doc, cut_all=False):
            if not word.strip():
                continue
            if word.strip() in self.stopwords_list:
                continue
            seg_text.append(word)
        return seg_text


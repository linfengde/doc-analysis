# 金融文档智能分析算法说明文档

创新实验室	林越峰

2019年6月13日

## 项目说明

本项目实现了智能文档分析，提供以下8个功能点

```
******* 功能点 ********
1.近义词词频统计              word_count
2.共现词发现                  concurrence_word_count
3.比较2篇文章新增词           newword_count
4.比较2篇文章相同词变化趋势   sameword_count
5.新词、短语发现功能          newword_found
6.差异化比较                  difference_count
7.查找关键词所在句子          find_sentences
8.生成文章关键词              generate_keywords
```

## 文件说明

`doc_analysis`:提供智能文档分析的8个功能点

## API说明

**1. 近义词词频统计**

```
word_count(input_str,count_only_singleword=False,calculate_tfidf_pseg=False,keywords_list=None)
```

```
:param input_txt:输入文档
:param count_only_singleword:  TRUE： 统计单个词     False: 近义词群
:param calculate_tfidf_pseg:  返回结果 是否包含词性，TFIDF值
:param keywords_list: 关键词列表
:return:   1.calculate_tfidf_pseg=TRUE    ['keys', 'value','tfidf', 'tag','length', 'weights']    # weight= value* tfidf
2.calculate_tfidf_pseg=False    ['keys', 'value']
```



**2.共现词发现**  

```
concurrence_word_count(input_txt,output_csv,threshold =4, write_csv=False,write_node_edge=False,keywords=None)
```



**3.比较2篇文章新增词**

```
 newword_count(compare_txt1,compare_txt2,output_csv)
```

```
:param compare_txt1: 分析文档
:param compare_txt2: 被比较文档
:param output_csv: 输出CSV路径
:return: dataframe       word value      关键词 出现次数
```



**4.比较2篇文章相同词变化趋势**   

```
sameword_count(compare_txt1,compare_txt2,output_csv)
```

```
:param compare_txt1: 分析文档
:param compare_txt2: 被比较文档
:param output_csv: 输出CSV路径
:return: dataframe       word value1 value2 diff         关键词 出现次数1 出现次数2    差值
```



**5.新词、短语发现功能**          

```
newword_found(input_str,mode='all',filter=True)
```

```
:param input_txt:分析文档 TXT格式
:param mode: 新词发现模式  all: 计算DOA,DOF，TFIDF,freq   part：只考虑TFIDF，freq
:param filter:
:return:
```



**6.差异化比较**                  

```
difference_count(compare_txt1,compare_txt2,output_csv,output_csv2,write_csv=False,count_concurrence_diff=False)
```

```
:param compare_txt1:被比较文章
:param compare_txt2:主要分析文章
:param output_csv: 输出路径
:param write_csv: 是否写入CSV文件
:return:
```



**7.查找关键词所在句子**          

```
find_sentences(keywords_list, input_str, topn)
```



**8.生成文章关键词**              

```
generate_keywords(input_txt, output_csv,write_csv=True)
```


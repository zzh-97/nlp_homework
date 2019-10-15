
## 基础理论部分

> **评阅点**：每道题是否回答完整

#### 0. Can you come up out 3 sceneraies which use AI methods? 

Ans: Search engine, Voice assistant, Image recognition.

#### 1. How do we use Github; Why do we use Jupyter and Pycharm;

Ans: To manage and save code; We use Jupyter and Pycharm because they are convenient and easy.

#### 2. What's the Probability Model?

Ans: It describes the probability relationship between random variables.

#### 3. Can you came up with some sceneraies at which we could use Probability Model?

Ans: Recommendation system, machine translation and search engine .

#### 4. Why do we use probability and what's the difficult points for programming based on parsing and pattern match?

Ans: We use probability because it's more realistic, the difficult points for programming based on parsing and pattern match are that we cannot list all the patterns or find a pattern that suits every situation and it's trivial.

#### 5. What's the Language Model;

Ans: Language Model is a probability based model that can calculate the probability of a sentence.

#### 6. Can you came up with some sceneraies at which we could use Language Model?


Ans: Voice assistant, speech recognition.

#### 7. What's the 1-gram language model;

Ans: The probabilty of a sentence is equal to the product of the probabilities of each word in this sentence.

#### 8. What's the disadvantages and advantages of 1-gram language model;

Ans: disadvantage: unrealistic. advantages: quicker and needs less memory.

#### 9. What't the 2-gram models;

Ans:$$ Pr(sentence) = Pr(w_1 \cdot w_2 \cdots w_n) = \prod \frac{count(w_i, w_{i+1})}{count(w_i)}$$

## 编程实践部分

#### 1. 设计你自己的句子生成器

如何生成句子是一个很经典的问题，从1940s开始，图灵提出机器智能的时候，就使用的是人类能不能流畅和计算机进行对话。和计算机对话的一个前提是，计算机能够生成语言。

计算机如何能生成语言是一个经典但是又很复杂的问题。 我们课程上为大家介绍的是一种基于规则（Rule Based）的生成方法。该方法虽然提出的时间早，但是现在依然在很多地方能够大显身手。值得说明的是，现在很多很实用的算法，都是很久之前提出的，例如，二分查找提出与1940s, Dijstra算法提出于1960s 等等。

在著名的电视剧，电影《西部世界》中，这些机器人们语言生成的方法就是使用的SyntaxTree生成语言的方法。



在这一部分，需要各位同学首先定义自己的语言。 大家可以先想一个应用场景，然后在这个场景下，定义语法。例如：

在西部世界里，一个”人类“的语言可以定义为：
``` 
human = """
human = 自己 寻找 活动
自己 = 我 | 俺 | 我们 
寻找 = 看看 | 找找 | 想找点
活动 = 乐子 | 玩的
"""
```

一个“接待员”的语言可以定义为
```
host = """
host = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号 ,
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
玩玩 = 耍一耍 | 玩一玩
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？"""

```




请定义你自己的语法: 

第一个语法：


```python
librarian = '''
librarian = 寒暄 指代 报数 书籍 结尾
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 先生 | 女士 | 小朋友
打招呼 = 你好, | 您好, | 下午好,
指代 = 这 连接 | 那 连接
连接 = 是 | 有
报数 = 数字 本
数字 = 1 | 2 | 3 | 4
书籍 = 小说 | 字典 | 杂志 | 期刊 
结尾 = .'''
```

> **评阅点**： 是否提出了和课程上区别较大的语法结构

第二个语法：


```python
praise = '''
praise = 人称 感叹 形容词 结尾
人称 = 单数 | 复数
单数 = 我 | 您 | 你 | 他 | 她 | 它
复数 = 我们 | 你们 | 他们
感叹 = 真 | 太 | 非常 | 超级
形容词 = 优秀 | 形容词 形容词*
形容词* = 美丽 | 善良 | 可爱 | 努力 | 聪明 | 帅气 
结尾 = 呀！| 啦 |. '''
```

> **评阅点**：是否和上一个语法区别比较大

TODO: 然后，使用自己之前定义的generate函数，使用此函数生成句子。


```python
def create_grammar(grammar_str, split='=', line_split='\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip(): continue  
        exp, stmt = line.split(split)
        grammar[exp.strip()] =[s.split() for s in stmt.split('|')] 
    return grammar
```


```python
import random
choice = random.choice
def generate(gram, target):
    if target not in gram: return target 
    a=choice(gram[target])
    expaned = [generate(gram, t) for t in a]
    return ''.join([e if e != '/n' else '\n' for e in expaned if e != 'null'])
```


```python
generate(create_grammar(librarian),'librarian')
```




    '下午好,那是2本期刊.'




```python
generate(create_grammar(praise),'praise')
```




    '我们太优秀.'



TODO: 然后，定义一个函数，generate_n，将generate扩展，使其能够生成n个句子:


```python
def generate_n(n):
    for i in range(n):
        print(generate(create_grammar(librarian),'librarian'))
```


```python
generate_n(3)
```

    你好,那是3本小说.
    你好,这有2本字典.
    下午好,这是3本字典.


> **评阅点**; 运行代码，观察是否能够生成多个句子

#### 2. 使用新数据源完成语言模型的训练

按照我们上文中定义的`prob_2`函数，我们更换一个文本数据源，获得新的Language Model:

1. 下载文本数据集（你可以在以下数据集中任选一个，也可以两个都使用）
    + 可选数据集1，保险行业问询对话集： https://github.com/Computing-Intelligence/insuranceqa-corpus-zh/raw/release/corpus/pool/train.txt.gz
    + 可选数据集2：豆瓣评论数据集：https://github.com/Computing-Intelligence/datasource/raw/master/movie_comments.csv
2. 修改代码，获得新的**2-gram**语言模型
    + 进行文本清洗，获得所有的纯文本
    + 将这些文本进行切词
    + 送入之前定义的语言模型中，判断文本的合理程度


```python
corpus='/Users/mac/movie_comments.csv'
```


```python
import pandas as pd
FILE=pd.read_csv(corpus)
```

    /anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3049: DtypeWarning: Columns (0,4) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
len(FILE['comment'])
```




    261497




```python
subf=FILE['comment'][:10000]
```


```python
str=''
for i in range(len(subf)-1):
    str=subf[i]+str
```


```python
len(str)
```




    499354




```python
import re
str = re.sub("[A-Za-z0-9\!\%\[\]\,\。\?\#\.\/\~\-\《\》\@\～]", "", str)
```


```python
len(str)
```




    460048




```python
import jieba
```


```python
def cut(string):
    return list(jieba.cut(string))
```


```python
TOKENS = cut(str)
```


```python
len(TOKENS)
```




    285619




```python
from collections import Counter
```


```python
words_count = Counter(TOKENS)
```


```python
_2_gram_words=[TOKENS[i]+TOKENS[i+1] for i in range(len(TOKENS)-1)]
```


```python
_2_gram_words_counts = Counter(_2_gram_words)
```


```python
_2_gram_words_counts.most_common()[-1]
```




    ('恶心想', 1)




```python
def get_1_gram_count(word):
    if word in words_count: return words_count[word]
    else:
        return words_count.most_common()[-1][-1]
```


```python
def get_2_gram_count(word):
    if word in _2_gram_word_counts: return _2_gram_word_counts[word]
    else:
        return _2_gram_word_counts.most_common()[-1][-1]
```


```python
def get_gram_count(word, wc):
    if word in wc: return wc[word]
    else:
        return wc.most_common()[-1][-1]
```


```python
def two_gram_model(sentence):     
    tokens=cut(sentence)
    prob=1
    for i in range(len(tokens)-1):
        word=tokens[i]
        next_word=tokens[i+1]
        _two_gram_c = get_gram_count(word+next_word, _2_gram_words_counts)
        _one_gram_c = get_gram_count(next_word, words_count)
        pro2 =  _two_gram_c / _one_gram_c
        
        prob*= pro2
    
    return prob
```


```python
two_gram_model(generate(create_grammar(librarian),'librarian'))
```




    6.021919788028422e-06



> **评阅点** 1. 是否使用了新的数据集； 2. csv(txt)数据是否正确解析

#### 3. 获得最优质的的语言

当我们能够生成随机的语言并且能判断之后，我们就可以生成更加合理的语言了。请定义 generate_best 函数，该函数输入一个语法 + 语言模型，能够生成**n**个句子，并能选择一个最合理的句子: 



提示，要实现这个函数，你需要Python的sorted函数


```python
sorted([1, 3, 5, 2])
```




    [1, 2, 3, 5]



这个函数接受一个参数key，这个参数接受一个函数作为输入，例如


```python
sorted([(2, 5), (1, 4), (5, 0), (4, 4)], key=lambda x: x[0])
```




    [(1, 4), (2, 5), (4, 4), (5, 0)]



能够让list按照第0个元素进行排序.


```python
sorted([(2, 5), (1, 4), (5, 0), (4, 4)], key=lambda x: x[1])
```




    [(5, 0), (1, 4), (4, 4), (2, 5)]



能够让list按照第1个元素进行排序.


```python
sorted([(2, 5), (1, 4), (5, 0), (4, 4)], key=lambda x: x[1], reverse=True)
```




    [(2, 5), (1, 4), (4, 4), (5, 0)]



能够让list按照第1个元素进行排序, 但是是递减的顺序。

>


```python
def generate_best(grammar,model,n):
    need_compared=[]
    for i in range(n):
        add=generate(create_grammar(grammar),list(create_grammar(grammar).keys())[0])
        need_compared.append(add)
    print(sorted(need_compared, key=lambda x: model(x))[0])
```


```python
generate_best(praise,two_gram_model,10)
```

    他超级优秀聪明帅气可爱美丽呀！


好了，现在我们实现了自己的第一个AI模型，这个模型能够生成比较接近于人类的语言。

> **评阅点**： 是否使用 lambda 语法进行排序

Q: 这个模型有什么问题？ 你准备如何提升？ 

Ans:数据集内容与语法内容关联度不高，增加服务业用语到现有数据集中。
变成 3-gram问题使模型更准确。
清洗后的数据仍保留标点或特殊符号，非纯文本。改用更规范的数据。

>**评阅点**: 是否提出了比较实际的问题，例如OOV问题，例如数据量，例如变成 3-gram问题。

##### 以下内容为可选部分，对于绝大多数同学，能完成以上的项目已经很优秀了，下边的内容如果你还有精力可以试试，但不是必须的。

#### 4. (Optional) 完成基于Pattern Match的语句问答
> 我们的GitHub仓库中，有一个assignment-01-optional-pattern-match，这个难度较大，感兴趣的同学可以挑战一下。


#### 5. (Optional) 完成阿兰图灵机器智能原始论文的阅读
1. 请阅读阿兰图灵关于机器智能的原始论文：https://github.com/Computing-Intelligence/References/blob/master/AI%20%26%20Machine%20Learning/Computer%20Machinery%20and%20Intelligence.pdf 
2. 并按照GitHub仓库中的论文阅读模板，填写完毕后发送给我: mqgao@kaikeba.com 谢谢

> 

各位同学，我们已经完成了自己的第一个AI模型，大家对人工智能可能已经有了一些感觉，人工智能的核心就是，我们如何设计一个模型、程序，在外部的输入变化的时候，我们的程序不变，依然能够解决问题。人工智能是一个很大的领域，目前大家所熟知的深度学习只是其中一小部分，之后也肯定会有更多的方法提出来，但是大家知道人工智能的目标，就知道了之后进步的方向。

然后，希望大家对AI不要有恐惧感，这个并不难，大家加油！

>

![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1561828422005&di=48d19c16afb6acc9180183a6116088ac&imgtype=0&src=http%3A%2F%2Fb-ssl.duitang.com%2Fuploads%2Fitem%2F201807%2F28%2F20180728150843_BECNF.thumb.224_0.jpeg)

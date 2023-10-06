---
title: Tokenizers in LLM
status: hidden
date: 2023-10-5 00:00
category: Blog
slug: tokenizers-llm
tags: machine learning, large language models
authors: Bishwa Karki
status: published
---

I'd like to compare large language models to a robot that can perform any task. Large language models have drawn people from a variety of fields, not just one. And this is a result of its attempts to become a perfect robot capable of doing the majority of tasks involving natural language.

Several researchers are now working on various areas of large language models to further improve their functionality. And in this article, I'll discuss tokenizer, a small but crucial part of a large language model.

## Tokenizer in LLM:

### Tokens:
We started learning not directly from sentence or words but from individual letters and gradually improved ourself to understand words and so on. It is also same for machines, they can't directly understand or process the sentence. Machines looks sentence into words and with the help of mechanism called attention they try to relate the meaning of each words in the sentence.

Thus, this units that the large language models can understand and process are called tokens. This tokens can be individual characters, words or subwords depending on the tokenization approach.

### Tokenizer:
Once the input sentence is given, the process of splitting sentence into tokens is called tokenization and it is done using tokenizer. Any model builder must take a close attention to tokenization process and tokenizer because it is key component. Tokenization not just breaks the sentence into words or part of words but it also helps in building vocabulary for large language models.

Different tokenizer and tokenization methods are used by todays large language models. Here we will see how it works to tokenize the sentence for some popular large language models and how they are different from each others.

We have this below text and pass to different tokenizer and see how the results vary.

```python
text = """
  English and TOKENIZATION
  😊 😁

  show_tokens TRUE None elif == >= else:
  Two tabs: "\t\t" Four spaces: "    "
  12.0*50=60
"""
```

#### BERT:
#### bert-base-uncased
This is the result after passing the above text to bert-base-uncased model.

```python
[CLS] english and capital ##ization [UNK] [UNK] show _ token ##s false none eli ##f = = > = else : two tab ##s : " " four spaces : " " 12 . 0 * 50 = 60 [SEP] 

```


Here we have the tokenized sentence and we can see [CLS] and [SEP] is being added to the sentence. Ans the new line in the sentence is being given as [UNK] and the emoji are also being taken as [UNK]. So we can see that some information is already being lost.
Also, what we can observe is that, the capital letters are not capitalized by the tokenizer which could also potentially loose some information. Capitalization may be useful in some task.


#### bert-base-cased

```python
[CLS] English and CA ##PI ##TA ##L ##I ##Z ##AT ##ION [UNK] [UNK] show _ token ##s F ##als ##e None el ##if = = > = else : Two ta ##bs : " " Four spaces : " " 12 . 0 * 50 = 60 [SEP] 
```

When we see the results from bert-base-cased model, one difference we can see is the capitalization is preserved here but we still have [UNK] for new lines.

### GPT-2:

Let's see the output tokenization from GPT-2 tokenizer.

```python
   English  and  CAP ITAL IZ ATION 
    � �  � � 

    show _ t ok ens  False  None  el if  ==  >=  else : 
    Two  tabs :  " 	 	 "  Four  spaces :  "        " 
    12 . 0 * 50 = 60 
```

Here, the new lines are being preserved and smoothly handles capitalization. The emoji's is handled with different tokens. And the other interesting thing is, we can reconstruct the emoji back using those tokens.

```python
def encode_decode(sentence, tokenizer_name):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  token_ids = tokenizer(sentence).input_ids
  print(tokenizer.decode(token_ids))

encode_decode(text, "gpt2")
```
```python
 English and CAPITALIZATION
  😊 😁

  show_tokens False None elif == >= else:
  Two tabs: "		" Four spaces: "    "
  12.0*50=60
```

### GPT-4:

GPT-4 tokenizer has quite similar output to GPT-2 but GPT-4 can be used for programming thus we can see how it handles elif part in the input text.

```python
   English  and  TOKEN IZATION 
    � �  � � 

    show _tokens  TRUE  None  elif  ==  >=  else :
    Two  tabs :  " 	 	 "  Four  spaces :  "      "
     12 . 0 * 50 = 60
```

We can also see how the spaces and tabs are taken differently. GPT-4 tokenizers takes indentation, tabs and spaces differently and uniquely tokenize. This is because GPT-4 works with coding problems and spaces matter in programming.


We saw a breif overview of how tokens are handled by different models. The numbers are handled differently because the numbers can be grow arbitrarily and tokenizers deals with them by broking the numbers so it can keep track of the numbers. 


These are only some of the examples but there are so much to unpack about tokenizer and tokenization process in large language models.
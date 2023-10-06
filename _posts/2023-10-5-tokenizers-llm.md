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

Different tokenizer and tokenization methods are used by todays large language models. Here we will see how it works to tokenize the sentence for some popular large language models.

"""
  English and TOKENIZATION
  😊 😁

  show_tokens TRUE None elif == >= else:
  Two tabs: "\t\t" Four spaces: "    "
  12.0*50=60
"""
This is the text, we will be working with.
#### BERT:
<img src="/assets/img/blogs/bert_base_tokenization.png" width="100%" />
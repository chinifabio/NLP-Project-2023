# NLP-Project-2023
## STS: Semantic Textual Similarity
- Website with data: http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark
- Dataset: The dataset consists of pairs of sentences that have been judged by humans on how
similar they are to one another. Here the similarity indicates the level of (semantic) agreement
between the meaning of the sentences. The sentences have been sourced from three domains
(news, caption and forum).
- Task: the original task here is to predict the similarity between the sentences. Note that this
problem could be handled as either a regression (predict a real score) or classification (predict
similar/non-similar label) task.
- Hints and suggestions: You could investigate various unsupervised techniques, such as
computing the cosine similarity between TF-IDF bag-of-words vectors for the sentences,
extending the BOW vector to include n-grams (bigrams and trigrams), and also consider the
distance between average word embeddings. Moreover, supervised models could be trained, in
particular transformer-based models are often fine-tuned to classify pairs of sentences. 

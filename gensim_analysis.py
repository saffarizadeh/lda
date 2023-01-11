# Train LDA model.
from collections import defaultdict
from nltk import word_tokenize
import pandas as pd
from lda.preprocessor import Preprocessor

df = pd.read_excel(
    '/Users/kaminem64/Library/CloudStorage/GoogleDrive-k.saffarizadeh@gmail.com/My Drive/Research/1 - Artificial Intelligence/2 - AI Indeterminacy in Conversational Agents/3 - Data/Experiment 5 - Car/UnstructuredData.xlsx')
preprocessor = Preprocessor()
cleaned_docs = preprocessor.clean_documents(df.qual.tolist())
tokenized_docs = [word_tokenize(doc) for doc in cleaned_docs]

# ------------------------------------------------------
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import ClippedCorpus
import tqdm

# Create a dictionary representation of the documents.
dictionary = Dictionary(tokenized_docs)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=2, no_above=0.5)
docs = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Set training parameters.
eval_every = None  # Don't evaluate model perplexity, takes too much time.
# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token


def compute_coherence_values(k):
    lda_model = LdaModel(corpus=docs,
                         id2word=id2word,
                         num_topics=k,
                         random_state=100,
                         chunksize=100,
                         passes=10,
                         iterations=400,
                         eval_every=None,
                         alpha='auto',
                         eta='auto')

    coherence_model_lda = CoherenceModel(model=lda_model, corpus=docs, dictionary=id2word, coherence='u_mass')

    return coherence_model_lda.get_coherence()


# coherence = []
# num_topics = []
# pbar = tqdm.tqdm(total=30)
# for k in range(1, 31):
#     num_topics.append(k)
#     coherence.append(compute_coherence_values(k=k))
#     pbar.update(1)
#
# pd.DataFrame({'Topics': num_topics, 'Coherence': coherence}).to_excel('lda_tuning_results.xlsx', index=False)
# pbar.close()


model = LdaModel(corpus=docs,
                 id2word=id2word,
                 num_topics=9,
                 random_state=100,
                 chunksize=100,
                 passes=10,
                 iterations=400,
                 eval_every=None,
                 alpha='auto',
                 eta='auto')

model.save('model9.gensim')

top_topics = model.top_topics(docs)

print(len(top_topics))
# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / 9
# print('Average topic coherence: %.4f.' % avg_topic_coherence)

word_scores = []
for topic in model.top_topics(docs):
    topic = topic[0]
    # topic is a list of tuples like [(0, 0.05888), ...]
    scores = list(zip(*topic))  # convert it to two lists
    word_scores.append(scores[1])

print(len(word_scores))
word_scores_df = pd.DataFrame(word_scores).T
word_scores_df.fillna(0, inplace=True)
word_scores_df.to_excel('word_scores_df.xlsx')

docs_scores = []
for doc in model.get_document_topics(docs):
    # doc is a list of tuples like [(0, 0.05888), ...]
    scores = list(zip(*doc))  # convert it to two lists
    docs_scores.append(scores[1])

docs_scores_df = pd.DataFrame(docs_scores)
docs_scores_df.fillna(0, inplace=True)
docs_scores_df.to_excel('docs_scores_df.xlsx')

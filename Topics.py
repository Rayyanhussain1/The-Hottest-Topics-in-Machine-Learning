import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

papers = pd.read_csv('datasets/papers.csv')
papers.head()
papers.drop(['id','event_type','pdf_name'],axis=1,inplace=True)
papers.head()
groups = papers.groupby('year')
counts = groups.size()
import matplotlib.pyplot
matplotlib ='inline'
counts.plot(kind='bar')

print(papers['title'].head())

papers['title_processed'] = papers['title'].map(lambda x: re.sub('[,\.!?]', '', x))

papers['title_processed'] = papers['title_processed'].map(lambda x: x.lower())

papers['title_processed'].head()

long_string  = ' '.join(papers['title_processed'])

wordcloud = wordcloud.WordCloud()

wordcloud.generate(long_string)

wordcloud.to_image()
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 

    plt.bar(x_pos, counts,align='center')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.title('10 most common words')
    plt.show()
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(papers['title_processed'])
plot_10_most_common_words(count_data, count_vectorizer)
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation as LDA
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
number_topics = 10
number_words = 10

lda = LDA(n_components=number_topics)
lda.fit(count_data)
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)
more_papers_published_in_2018 = True
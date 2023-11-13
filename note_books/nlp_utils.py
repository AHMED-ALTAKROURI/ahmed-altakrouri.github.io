from gensim.models import CoherenceModel
import matplotlib.patches as mpatches
from nltk.corpus import stopwords
from utils_li import *
from modeling_utils import *
import gensim
import matplotlib.pyplot as plt
import string
from nltk.stem import WordNetLemmatizer

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    # Convert a document to lowercase and split into words
    stop_free = [i for i in doc.lower().split() if i not in stop]

    # Remove punctuation characters
    punc_free = [ch for ch in stop_free if ch not in exclude]

    # Lemmatize words
    normalized = [lemma.lemmatize(word) for word in punc_free]

    return normalized


# Define function to compute coherence values for various number of topics
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values = []  # List to store coherence values
    model_list = []  # List to store LDA models

    # Iterate over the range of number of topics
    for num_topics in range(start, limit, step):
        # Create an LDA model with the current number of topics
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=50)
        model_list.append(model)  # Append the model to the list

        # Compute the coherence value using the C_V coherence measure
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())  # Append the coherence value to the list

    # Return the list of LDA models and coherence values
    return model_list, coherence_values


def extract_top_topic_and_prob(document_topics):
    if document_topics:
        # Get the topic with the highest probability
        top_topic_with_prob = max(document_topics, key=lambda item: item[1])
        return top_topic_with_prob  # Return the top topic and its probability
    else:
        return None  # If there are no document topics, return None


def apply_topic_extraction(df, column_name='document_topics'):
    # Apply the extract_top_topic_and_prob function to create new columns
    df['top_topic_with_prob'] = df[column_name].apply(extract_top_topic_and_prob)
    df['top_topic'] = df['top_topic_with_prob'].apply(lambda x: x[0] if x else None)
    df['top_topic_prob'] = df['top_topic_with_prob'].apply(lambda x: x[1] if x else None)
    return df


def clean(doc):
    # Convert a document to lowercase and split into words
    stop_free = [i for i in doc.lower().split() if i not in stop]

    # Remove punctuation characters
    punc_free = [ch for ch in stop_free if ch not in exclude]

    # Lemmatize words
    normalized = [lemma.lemmatize(word) for word in punc_free]

    return normalized


def generate_wordclouds(model, num_topics):
    # Generate and display word clouds for each topic
    for t in range(num_topics):
        plt.figure(figsize=(20, 20))
        plt.subplot(5, 4, t + 1)
        plt.title("Topic #" + str(t))
        topic_keywords = dict(model.show_topic(t, topn=10))
        cloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          prefer_horizontal=1.0)
        cloud.generate_from_frequencies(topic_keywords, max_font_size=300)
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_topics_tsne(model, corpus, num_topics):
    # Get the topic distributions for each document
    topic_distributions = np.array([model.get_document_topics(doc) for doc in corpus])

    # Create a matrix of topic probabilities
    topic_matrix = np.zeros((len(topic_distributions), num_topics))
    for i, doc in enumerate(topic_distributions):
        for topic, prob in doc:
            topic_matrix[i, topic] = prob

    # Use t-SNE to reduce dimensionality
    tsne_model = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=5000, verbose=1, random_state=0,
                      angle=0.5)
    tsne_vectors = tsne_model.fit_transform(topic_matrix)

    # Create a scatter plot of t-SNE vectors
    plt.figure(figsize=(10, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, num_topics))  # Create a colormap with `num_topics` colors
    plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=topic_matrix.argmax(axis=1), cmap='viridis', alpha=0.5)

    # Create a legend
    patches = [mpatches.Patch(color=colors[i], label='Topic #' + str(i)) for i in range(num_topics)]
    plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper left')

    plt.show()


def plot_topic_probabilities(df, width=800, height=600, nbins=50):
    # Create a histogram plot using Plotly Express
    fig = px.histogram(df, x='top_topic_prob', nbins=nbins, color="top_topic")

    # Update the layout of the plot
    fig.update_layout(
        title_text='Distribution of Top Topic Probabilities',
        xaxis_title='Probability',
        yaxis_title='Count',
        width=width,
        height=height
    )

    # Display the plot
    fig.show()

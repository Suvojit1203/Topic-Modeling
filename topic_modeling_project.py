
import os
import tarfile
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

def extract_dataset(archive_path, extract_to):
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)

def load_documents(base_path, max_docs=1000):
    documents = []
    labels = []
    categories = os.listdir(base_path)
    all_files = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                filepath = os.path.join(category_path, filename)
                all_files.append((filepath, category))

    random.shuffle(all_files)
    sampled_files = all_files[:max_docs]

    for filepath, category in sampled_files:
        try:
            with open(filepath, 'r', errors='ignore') as file:
                content = file.read()
                documents.append(content)
                labels.append(category)
        except:
            continue

    return documents, labels

def get_top_terms(model_components, terms, n_top_words=10):
    topics = []
    for topic_weights in model_components:
        top_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
        topics.append([terms[i] for i in top_indices])
    return topics

def main():
    dataset_archive = "20_newsgroups.tar.gz"
    dataset_folder = "20_newsgroups"
    if not os.path.exists(dataset_folder):
        print("Extracting dataset...")
        extract_dataset(dataset_archive, dataset_folder)

    base_path = os.path.join(dataset_folder, "20_newsgroups")
    print("Loading documents...")
    documents, labels = load_documents(base_path)

    print("Vectorizing with TF-IDF for K-means...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    print("Vectorizing with CountVectorizer for LDA...")
    count_vectorizer = CountVectorizer(stop_words='english', max_df=0.5, min_df=5)
    count_matrix = count_vectorizer.fit_transform(documents)

    num_clusters = 10

    print("Applying K-means clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    kmeans_terms = get_top_terms(kmeans.cluster_centers_, tfidf_vectorizer.get_feature_names_out())

    print("\nTop terms per K-means cluster:")
    for i, terms in enumerate(kmeans_terms):
        print(f"Cluster {i + 1}: {', '.join(terms)}")

    print("\nApplying LDA topic modeling...")
    lda = LatentDirichletAllocation(n_components=num_clusters, random_state=42)
    lda.fit(count_matrix)
    lda_terms = get_top_terms(lda.components_, count_vectorizer.get_feature_names_out())

    print("\nTop terms per LDA topic:")
    for i, terms in enumerate(lda_terms):
        print(f"Topic {i + 1}: {', '.join(terms)}")

if __name__ == "__main__":
    main()

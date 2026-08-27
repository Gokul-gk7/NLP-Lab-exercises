from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ---------------------------------------------------
# EXPERIMENT NO. 10
# Social Media Clustering using TF-IDF and K-Means
# ---------------------------------------------------

print("=" * 60)
print("SOCIAL MEDIA POST CLUSTERING")
print("TF-IDF + K-MEANS")
print("=" * 60)


# ---------------------------------------------------
# Step 1: Accept social media posts
# ---------------------------------------------------

posts = []

n = int(input("\nEnter number of posts: "))

if n < 2:
    print("Please enter at least 2 posts.")
    exit()

for i in range(n):

    post = input(f"Enter post {i + 1}: ")

    posts.append(post.lower())


# ---------------------------------------------------
# Step 2: Accept number of clusters
# ---------------------------------------------------

k = int(input("\nEnter number of clusters: "))

if k < 2 or k > n:
    print("Number of clusters must be between 2 and number of posts.")
    exit()


# ---------------------------------------------------
# Step 3: TF-IDF Vectorization
# Unigrams + Bigrams
# ---------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(posts)


print("\nTF-IDF vectorization completed successfully.")


# ---------------------------------------------------
# Step 4: Apply K-Means clustering
# ---------------------------------------------------

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

model.fit(X)


# ---------------------------------------------------
# Step 5: Get cluster labels
# ---------------------------------------------------

labels = model.labels_


# ---------------------------------------------------
# Step 6: Display clustered posts
# ---------------------------------------------------

print("\n" + "=" * 50)
print("CLUSTER RESULTS")
print("=" * 50)

for cluster in range(k):

    print(f"\n--- Cluster {cluster + 1} ---")

    for i in range(n):

        if labels[i] == cluster:

            print(
                f"Post {i + 1}: {posts[i]}"
            )


# ---------------------------------------------------
# Step 7: Extract important keywords and phrases
# ---------------------------------------------------

terms = vectorizer.get_feature_names_out()

print("\n" + "=" * 50)
print("IMPORTANT KEYWORDS AND PHRASES")
print("=" * 50)

for cluster in range(k):

    center = model.cluster_centers_[cluster]

    top_indices = center.argsort()[-5:][::-1]

    print(f"\nCluster {cluster + 1}:")

    for index in top_indices:

        print(
            f"- {terms[index]}"
        )


# ---------------------------------------------------
# Step 8: Display marketing insights
# ---------------------------------------------------

print("\n" + "=" * 50)
print("MARKETING INSIGHTS")
print("=" * 50)

for cluster in range(k):

    center = model.cluster_centers_[cluster]

    top_indices = center.argsort()[-3:][::-1]

    keywords = [
        terms[index]
        for index in top_indices
    ]

    print(
        f"\nCluster {cluster + 1} focuses on: "
        + ", ".join(keywords)
    )

    print(
        "Marketing Insight: This cluster represents "
        "a group of customers with similar interests "
        "or opinions."
    )


# ---------------------------------------------------
# Step 9: Final conclusion
# ---------------------------------------------------

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(
    "Social media posts were successfully clustered "
    "using TF-IDF and K-Means."
)

print(
    "The clusters revealed customer interests, "
    "trends and potential marketing opportunities."
)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE


# ---------------------------------------------------
# EXPERIMENT NO. 8
# Topic Modeling using LDA and Visualization using t-SNE
# ---------------------------------------------------

print("=" * 60)
print("TOPIC MODELING OF CUSTOMER REVIEWS")
print("LDA + t-SNE")
print("=" * 60)

# Step 1: Accept customer reviews
reviews = []

n = int(input("\nEnter number of reviews (minimum 3): "))

if n < 3:
    print("Please enter at least 3 reviews.")
    exit()

for i in range(n):
    review = input(f"Enter review {i + 1}: ")
    reviews.append(review)


# Step 2: Convert reviews into numerical vectors
vectorizer = CountVectorizer(stop_words="english")

X = vectorizer.fit_transform(reviews)

print("\nReviews converted into numerical vectors successfully.")


# Step 3: Apply LDA
lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)


# Step 4: Display important keywords for each topic
words = vectorizer.get_feature_names_out()

print("\n" + "=" * 40)
print("EXTRACTED TOPICS")
print("=" * 40)

for i, topic in enumerate(lda.components_):

    print(f"\nTopic {i + 1}:")

    top_words = topic.argsort()[-5:][::-1]

    for j in top_words:
        print(words[j])


# Step 5: Get topic distribution for each review
topic_distribution = lda.transform(X)

print("\n" + "=" * 40)
print("TOPIC DISTRIBUTION")
print("=" * 40)

for i, distribution in enumerate(topic_distribution):

    print(
        f"Review {i + 1}: "
        f"Topic 1 = {distribution[0]:.3f}, "
        f"Topic 2 = {distribution[1]:.3f}"
    )


# Step 6: Apply t-SNE
X_dense = X.toarray()

# Perplexity must be smaller than number of samples
perplexity_value = min(2, n - 1)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=perplexity_value
)

X_tsne = tsne.fit_transform(X_dense)


# Step 7: Display t-SNE coordinates
print("\n" + "=" * 40)
print("t-SNE COORDINATES")
print("=" * 40)

for i, point in enumerate(X_tsne):

    print(
        f"Review {i + 1}: "
        f"X = {point[0]:.4f}, "
        f"Y = {point[1]:.4f}"
    )


# Step 8: Visualize review clusters
plt.figure(figsize=(10, 7))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    s=100
)

# Add review labels
for i in range(len(reviews)):

    plt.text(
        X_tsne[i, 0],
        X_tsne[i, 1],
        "R" + str(i + 1),
        fontsize=12
    )

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)

plt.show()


# Step 9: Result
print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(
    "Topic modeling successfully extracted important themes "
    "from customer reviews."
)

print(
    "t-SNE visualization represented the reviews in a "
    "two-dimensional space."
)
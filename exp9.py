from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ---------------------------------------------------
# EXPERIMENT NO. 9
# Rule-Based Classifier vs Maximum Entropy Classifier
# ---------------------------------------------------

print("=" * 60)
print("LEGAL DOCUMENT CLASSIFICATION")
print("RULE-BASED vs MAXIMUM ENTROPY CLASSIFIER")
print("=" * 60)


# ---------------------------------------------------
# Step 1: Accept documents and categories
# ---------------------------------------------------

docs = []
labels = []

n = int(input("\nEnter number of documents: "))

if n < 3:
    print("Please enter at least 3 documents.")
    exit()

for i in range(n):

    document = input(f"\nEnter document {i + 1}: ")
    category = input(
        "Enter category (contract/judgment/agreement): "
    )

    docs.append(document)
    labels.append(category.lower())


# ---------------------------------------------------
# Step 2: Rule-Based Classification
# ---------------------------------------------------

rule_pred = []

for doc in docs:

    doc_lower = doc.lower()

    if "contract" in doc_lower:
        rule_pred.append("contract")

    elif "judgment" in doc_lower:
        rule_pred.append("judgment")

    elif "agreement" in doc_lower:
        rule_pred.append("agreement")

    else:
        # Default category
        rule_pred.append("agreement")


# ---------------------------------------------------
# Step 3: Calculate Rule-Based Accuracy
# ---------------------------------------------------

rule_acc = accuracy_score(labels, rule_pred)


# ---------------------------------------------------
# Step 4: Convert documents into vectors
# ---------------------------------------------------

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(docs)


# ---------------------------------------------------
# Step 5: Maximum Entropy Classifier
# Logistic Regression implements a MaxEnt model
# ---------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, labels)


# ---------------------------------------------------
# Step 6: Predict categories
# ---------------------------------------------------

ml_pred = model.predict(X)


# ---------------------------------------------------
# Step 7: Calculate Maximum Entropy Accuracy
# ---------------------------------------------------

ml_acc = accuracy_score(labels, ml_pred)


# ---------------------------------------------------
# Step 8: Display predictions
# ---------------------------------------------------

print("\n" + "=" * 50)
print("CLASSIFICATION RESULTS")
print("=" * 50)

for i in range(n):

    print(f"\nDocument {i + 1}")
    print("Actual Category       :", labels[i])
    print("Rule-Based Prediction :", rule_pred[i])
    print("MaxEnt Prediction     :", ml_pred[i])


# ---------------------------------------------------
# Step 9: Display accuracy
# ---------------------------------------------------

print("\n" + "=" * 50)
print("ACCURACY COMPARISON")
print("=" * 50)

print(
    f"Rule-Based Accuracy     : {rule_acc * 100:.2f}%"
)

print(
    f"Maximum Entropy Accuracy: {ml_acc * 100:.2f}%"
)


# ---------------------------------------------------
# Step 10: Comparison
# ---------------------------------------------------

print("\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)

if rule_acc > ml_acc:

    print("Rule-Based Classifier performed better.")

elif ml_acc > rule_acc:

    print("Maximum Entropy Classifier performed better.")

else:

    print("Both classifiers achieved the same accuracy.")


print("\nResult:")
print(
    "Legal documents were successfully classified using "
    "Rule-Based and Maximum Entropy classifiers."
)

print(
    "The classification accuracies were calculated and compared."
)
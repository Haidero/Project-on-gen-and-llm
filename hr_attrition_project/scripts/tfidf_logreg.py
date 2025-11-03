# scripts/tfidf_logreg.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 2. Create a text version of structured data
def row_to_text(row):
    return f"{row['JobRole']} in {row['Department']} with {row['YearsAtCompany']} years and performance rating {row['PerformanceRating']}."
df['text'] = df.apply(row_to_text, axis=1)

# 3. Encode target
df['label'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], stratify=df['label'], random_state=42
)

# 5. TF-IDF vectorizer + Logistic Regression
vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = LogisticRegression(max_iter=2000, class_weight='balanced')
clf.fit(Xtr, y_train)

# 6. Evaluate model
y_pred = clf.predict(Xte)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))

# 7. Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(clf, Xte, y_test)
plt.title("TF-IDF + Logistic Regression")
plt.show()


# 8. Save report to results file
report = classification_report(y_test, y_pred, digits=4)
with open("results/metrics.txt", "a") as f:
    f.write("\nTF-IDF + Logistic Regression\n")
    f.write(report + "\n")

print("✅ Results appended to results/metrics.txt")

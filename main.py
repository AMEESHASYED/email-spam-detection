import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load and clean the data
df = pd.read_csv("emails_V2.csv", encoding="latin-1")
df = df.dropna(subset=['text', 'spam'])
df['spam'] = df['spam'].astype(int)

print(df.shape)
print(df['spam'].value_counts())

# Preprocess the text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = text.replace("subject:", "")
    text = re.sub(r'[^a-z\s]', '', text)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)



# train the data
# Split, vectorize, train
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['spam'],
    test_size=0.2, stratify=df['spam'], random_state=42)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

def predict_email(msg):
    cleaned = clean_text(msg)
    vec = tfidf.transform([cleaned])
    return "Spam" if model.predict(vec)[0] == 1 else "Not Spam"

print(predict_email("Congratulations! You won a free prize. Click here to claim now"))
print(predict_email("Hi team, the meeting is moved to 3 pm tomorrow"))
print(predict_email("Subject: get cheap software cds, limited offer, buy now"))
print(predict_email("Please find attached the project report for review"))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "SVM": LinearSVC(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
}

results = []
for name, m in models.items():
    m.fit(X_train_tfidf, y_train)
    pred = m.predict(X_test_tfidf)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "Precision (spam)": round(precision_score(y_test, pred), 3),
        "Recall (spam)": round(recall_score(y_test, pred), 3),
        "Recall (not spam)": round(recall_score(y_test, pred, pos_label=0), 3),
        "F1 (spam)": round(f1_score(y_test, pred), 3),
    })
# step6
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC(class_weight='balanced')),
])

param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df': [1, 2],
    'clf__C': [0.1, 1, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=cv)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation F1 (macro):", round(grid.best_score_, 3))

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred_best), 3))
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))

scores = cross_val_score(best_model, df['clean_text'], df['spam'], cv=cv, scoring='accuracy')
print("CV accuracy per fold:", scores.round(3))
print("Mean CV accuracy:", round(scores.mean(), 3), "+/-", round(scores.std(), 3))

# final step
import joblib
joblib.dump(best_model, "spam_model.pkl")
print("Model saved")
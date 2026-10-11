# Email Spam Detection

A machine learning project that classifies an email as **Spam** or **Not Spam**, with a small web app (Streamlit) where you can paste an email and see the result.

## Results

| Metric | Value |
|---|---|
| Final model | TF-IDF (word pairs) + LinearSVC (SVM) |
| Test accuracy | 99.5% |
| Cross-validation accuracy (5-fold) | 99.1% ± 0.4% |
| Normal emails wrongly marked as spam | 2 of 135 |
| Spam emails missed | 0 of 274 |

Confusion matrix on the test set (409 emails):

|  | Predicted Not Spam | Predicted Spam |
|---|---|---|
| **Actual Not Spam** | 133 | 2 |
| **Actual Spam** | 0 | 274 |

### Model comparison (first test split, TF-IDF single words)

| Model | Accuracy | Precision (spam) | Recall (spam) | Recall (not spam) | F1 (spam) |
|---|---|---|---|---|---|
| Naive Bayes (baseline) | 0.963 | 0.948 | 1.000 | 0.889 | 0.973 |
| Logistic Regression | 0.988 | 0.982 | 1.000 | 0.963 | 0.991 |
| SVM (LinearSVC) | 0.990 | 0.986 | 1.000 | 0.970 | 0.993 |
| Random Forest | 0.980 | 0.982 | 0.989 | 0.963 | 0.985 |

The SVM was then tuned with GridSearchCV (best settings: `C=1`, `min_df=2`, `ngram_range=(1, 2)`), which cut the wrongly flagged normal emails from 15 (Naive Bayes) to 2.

## Dataset

- File: `emails_V2.csv`, with columns `text` (the email) and `spam` (1 = spam, 0 = not spam).
- 5,726 rows in the file, of which 2,043 had values. Empty rows were removed.
- After cleaning: 1,368 spam and 675 not spam emails.
- Split: 80% training / 20% testing, stratified so both sets keep the same spam ratio.

## Method

1. **Clean the data:** drop empty rows, convert labels to integers.
2. **Preprocess the text:** lowercase, remove "Subject:", punctuation and numbers, remove stopwords, apply stemming (NLTK).
3. **Features:** TF-IDF, using single words and word pairs (n-grams 1 to 2).
4. **Models compared:** Naive Bayes, Logistic Regression, LinearSVC, Random Forest (class weights balanced where supported).
5. **Tuning:** GridSearchCV with 5-fold stratified cross-validation, scored by macro F1.
6. **Evaluation:** accuracy, precision, recall, F1, confusion matrix, cross-validation.
7. **Deployment:** the final model is saved with `joblib` and used by a Streamlit app.

## Project structure

```
PythonProject6/
├── main.py            # data cleaning, training, comparison, tuning, saves the model
├── app.py             # Streamlit web app
├── nltk_setup.py      # one-time download of NLTK data
├── emails_V2.csv      # dataset
├── spam_model.pkl     # saved model (created by main.py)
├── requirements.txt   # libraries
└── README.md
```

## How to run

1. Install the libraries:
   ```
   pip install pandas numpy scikit-learn nltk matplotlib seaborn streamlit joblib
   ```
2. Download the NLTK data once (run `nltk_setup.py`, or):
   ```python
   import nltk
   nltk.download('stopwords')
   ```
3. Train the model and create `spam_model.pkl`:
   ```
   python main.py
   ```
4. Start the web app (use this command, not the Run button):
   ```
   streamlit run app.py
   ```
5. Open `http://localhost:8501`, paste an email, and click **Check**.

## Tech stack

Python 3.12, pandas, scikit-learn, NLTK, matplotlib, seaborn, Streamlit, joblib.

## Limitations and future work

- The dataset is small (about 2,000 emails), so results may be lower on emails that look different from this data.
- The confidence score in the app is approximate unless the model is calibrated.
- Ideas: train on a larger dataset such as Enron or SpamAssassin, try deep learning models (for example BERT), and detect links and attachments as extra features.

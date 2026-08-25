\# Fake News Detection Using NLP



\## Overview



This project develops a machine learning pipeline for classifying news

articles as \*\*True\*\* or \*\*Fake\*\* using Natural Language Processing (NLP).



The project uses TF-IDF-based text representations with unigram and bigram

features and compares three supervised classification algorithms:



\- Logistic Regression

\- Multinomial Naive Bayes

\- Linear Support Vector Machine (SVM)



The objective is to investigate how effectively textual information can be

used to distinguish between the two classes and to compare the performance

of different machine learning approaches.



\---



\## Dataset



The dataset consists of two collections of news articles:



\- `true.csv` — articles labelled as True

\- `fake.csv` — articles labelled as Fake



Both datasets contain the following variables:



| Variable | Description |

|---|---|

| `title` | News article title |

| `text` | Full article text |

| `subject` | News category/subject |

| `date` | Publication date |



The original data contained 21,417 true articles and 23,481 fake articles.



Before modelling, duplicate article texts were removed, resulting in

38,646 unique articles.



The final dataset contained:



\- 21,191 True articles

\- 17,455 Fake articles



The data contained no missing values.



> The original CSV files are not included in this repository. See

> `data/README.md` for information about preparing the data locally.



\---



\## Methodology



\### 1. Data Preparation



The two datasets were assigned binary labels:



\- `0` → True

\- `1` → Fake



The datasets were then combined and randomly shuffled.



Duplicate article texts were removed to reduce the possibility of the same

article appearing in both training and testing data.



The title and article text were concatenated to form the text input used for

classification.



\### 2. Train-Test Split



The data was divided into:



\- 80% training data

\- 20% testing data



A stratified split was used to preserve the class distribution between the

training and test sets.



\### 3. TF-IDF Feature Extraction



The textual data was converted into numerical features using

\*\*Term Frequency-Inverse Document Frequency (TF-IDF)\*\*.



The representation used:



\- Maximum 5,000 features

\- Unigrams

\- Bigrams



The TF-IDF vectorizer was fitted only on the training data and subsequently

used to transform the test data.



\### 4. Classification Models



Three classification algorithms were compared:



\#### Logistic Regression



A linear probabilistic classification model trained on the TF-IDF features.



\#### Multinomial Naive Bayes



A probabilistic classifier commonly used for text classification.



\#### Linear SVM



A linear Support Vector Machine trained on the TF-IDF representation.



\---



\## Results



Performance was evaluated on the held-out test set using accuracy,

precision, recall, and F1-score.



| Model | Accuracy | Precision | Recall | F1-score |

|---|---:|---:|---:|---:|

| Logistic Regression | 98.87% | 99.22% | 98.28% | 98.75% |

| Multinomial Naive Bayes | 96.31% | 96.36% | 95.45% | 95.90% |

| \*\*Linear SVM\*\* | \*\*99.42%\*\* | \*\*99.43%\*\* | \*\*99.28%\*\* | \*\*99.36%\*\* |



Linear SVM achieved the best performance across all four reported metrics.



\### Best Model: Linear SVM



The Linear SVM produced the following confusion matrix on the test set:



| | Predicted True | Predicted Fake |

|---|---:|---:|

| Actual True | 4,219 | 20 |

| Actual Fake | 25 | 3,466 |



This corresponds to 45 misclassified articles out of 7,730 test articles.



\---



\## Feature Interpretation



The coefficients of the Linear SVM were examined to identify features

strongly associated with each class.



Features associated with the Fake class included terms such as:



\- `via`

\- `read more`

\- `video`

\- `president trump`

\- `breaking`

\- `featured image`

\- `getty images`



Features strongly associated with the True class included:



\- `reuters`

\- `washington reuters`

\- `said`

\- `reuters the`

\- `president donald`



This analysis suggests that the classifier learned not only linguistic

patterns but also source- and formatting-related characteristics present

in the dataset.



\---



\## Limitations



The high test-set performance should not be interpreted as evidence that

the model will achieve the same performance on arbitrary real-world news.



Feature analysis indicates that some highly predictive features are related

to source attribution and article formatting. For example, `Reuters` was

strongly associated with the True class, while phrases such as `read more`

and `featured image` were associated with the Fake class.



Therefore, the model may partly exploit dataset-specific patterns rather

than relying solely on the factual content of an article.



Future work could investigate the model's robustness on articles from

previously unseen sources and datasets.



\---



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Jupyter Notebook



\---



\## Project Structure



```text

fake-news-detection/

│

├── README.md

├── Fake\_News\_Detection.ipynb

├── requirements.txt

│

├── data/

│   └── README.md

│

└── visuals/

&#x20;   ├── logistic\_regression\_confusion\_matrix.png

&#x20;   ├── svm\_confusion\_matrix.png

&#x20;   └── model\_comparison.png


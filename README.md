# 📰 Fake News Detection using GloVe Embeddings, WordNet Lemmatization & Hybrid Ensemble Learning

A visualization-driven, semantically-enriched machine learning framework that classifies news articles as **Fake** or **Real**, combining classical NLP preprocessing, GloVe word embeddings, and a hybrid ensemble of six classifiers — backed by a full research paper.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20WordNet-yellowgreen)
![Embeddings](https://img.shields.io/badge/Embeddings-GloVe%20100d-orange)
![ML](https://img.shields.io/badge/ML-Scikit--learn-red)
![Status](https://img.shields.io/badge/Status-Research%20Backed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Misinformation spreads faster than ever across digital platforms, and manually verifying every news article is no longer feasible. This project builds an **interpretable, semantically-aware fake news detection system** that goes beyond standard TF-IDF/Bag-of-Words pipelines by integrating **GloVe (Global Vectors for Word Representation)** embeddings and **WordNet-based part-of-speech lemmatization**, then evaluates six machine learning classifiers combined into a **majority-voting hybrid ensemble**.

This work is backed by a **full research paper** (included in this repo) authored as part of a Data Handling and Visualization project at **Netaji Subhas University of Technology (NSUT)**, co-authored with Kamran Ahmed and Ajisth Shukla.

## 🎯 Why This Project Stands Out

- **Beyond frequency-based features:** Most fake news classifiers stop at TF-IDF/CountVectorizer. This project adds **100-dimensional GloVe embeddings** to capture deep semantic relationships between words that pure frequency counts miss.
- **Linguistically-informed preprocessing:** Instead of naive stemming, the pipeline uses **POS-tag-aware WordNet lemmatization**, preserving grammatical meaning while reducing vocabulary noise.
- **Six classifiers, one robust ensemble:** Naïve Bayes, SVM, Random Forest, Decision Tree, KNN, and Logistic Regression are each trained and cross-validated individually, then fused via **majority voting** — reducing individual model bias and improving generalization.
- **Visualization as a first-class citizen:** Class distributions, text-length histograms, word clouds, and confusion matrices are used throughout to make model behavior transparent and interpretable, not just a black-box accuracy number.
- **Research-paper backed:** The methodology, novelty, and results are formally documented and written up in academic paper format — demonstrating the ability to both build and communicate ML work rigorously.

## 🧠 Methodology

### 1. Data
- **Source:** Kaggle's [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Attributes:** `id`, `title`, `text`, `subject`, `date`, `label`
- A balanced set of labeled fake and real news articles, with a held-out manual test set (10 fake + 10 real samples) reserved to check real-world generalization beyond the standard train/test split.

### 2. Preprocessing Pipeline
- Lowercasing, and removal of HTML tags, URLs, punctuation, and numeric tokens
- Tokenization and stopword removal using NLTK
- **POS-tag-based WordNet lemmatization** — words are lemmatized according to their grammatical role (noun, verb, adjective, etc.) rather than blindly stripped, preserving semantic accuracy
- 80/20 stratified train-test split, with a separate manual validation set

### 3. Feature Engineering
- **CountVectorizer** (unigrams + bigrams) for Bag-of-Words representation
- **TF-IDF** for discriminative, frequency-weighted features (used for the Naïve Bayes baseline)
- **GloVe embeddings (100-dim):** each article is converted into a dense semantic vector by averaging the GloVe vectors of its constituent words — capturing contextual meaning that frequency-based methods cannot

### 4. Exploratory Data Analysis
- Class distribution plots confirming a near-balanced dataset
- Text length and word count histograms/box plots — revealing that fake news tends toward shorter, more emotionally charged language, while real news is longer and more structurally consistent
- Comparative word clouds — fake news skews toward sensational/subjective terms; real news skews toward factual, domain-specific vocabulary

### 5. Models Trained
| Model | Feature Input | Role |
|---|---|---|
| Multinomial Naïve Bayes | TF-IDF | Probabilistic baseline |
| Linear SVM (LinearSVC) | GloVe | High-dimensional margin classifier |
| Random Forest | GloVe | Non-linear ensemble, feature importance |
| Decision Tree | GloVe | Interpretable rule-based model |
| K-Nearest Neighbors | GloVe | Distance-based baseline |
| Logistic Regression | GloVe | Linear, interpretable baseline |
| **Hybrid Ensemble** | All of the above | **Majority-voted final classifier** |

### 6. Evaluation
Each model was evaluated on Accuracy, Precision, Recall, and F1-score on the held-out test set, with confusion matrices generated for every classifier to visualize error types (false positives vs. false negatives).

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Naïve Bayes | 0.7206 | 0.9713 | 0.4269 | 0.5931 |
| SVM | 0.9313 | 0.9269 | 0.9292 | 0.9280 |
| Random Forest | 0.9414 | 0.9448 | 0.9316 | 0.9381 |
| Decision Tree | 0.8924 | 0.9032 | 0.8674 | 0.8849 |
| KNN | 0.9151 | 0.8897 | 0.9383 | 0.9134 |
| Logistic Regression | 0.9269 | 0.9252 | 0.9274 | 0.9263 |
| **Hybrid Ensemble** | **0.9433** | **0.9692** | **0.9101** | **0.9387** |

The hybrid ensemble correctly classified 4,570 of 4,694 fake news articles and 3,897 of 4,282 real news articles, with only 124 false positives and 385 false negatives — confirming strong diagonal dominance and low Type I/Type II error rates in the confusion matrix.

*(Full metric derivations, confusion matrix visualizations, word clouds, and performance plots are available in the notebook and the accompanying research paper.)*

## 🔬 Key Novelty (from the accompanying research paper)

1. **Advanced text preprocessing** — POS-based WordNet lemmatization instead of naive stemming/tokenization
2. **Semantic feature enrichment** — GloVe embeddings layered on top of traditional vectorization
3. **Hybrid ensemble via majority voting** — combining probabilistic, distance-based, and tree-based paradigms into one robust classifier
4. **Visualization-driven interpretability** — word clouds, histograms, and confusion matrices used throughout to explain *why* the model works, not just *that* it works
5. **Reproducibility pipeline** — evaluation metrics computed and exported systematically, with model/preprocessing artifacts intended for serialization (`joblib`/`pickle`) for deployment readiness

## 🛠️ Tech Stack

- **Language:** Python
- **Environment:** Google Colab
- **NLP:** NLTK, WordNet, POS tagging
- **Embeddings:** GloVe (glove.6B.100d)
- **ML:** scikit-learn (Naïve Bayes, SVM, Random Forest, Decision Tree, KNN, Logistic Regression)
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Data Handling:** Pandas, NumPy

## 📁 Repository Structure

```
Fake-News-Detection/
├── Fake_News_Detection_GloVe_WordNet.ipynb   # Full Colab pipeline: preprocessing → GloVe → 6 models → ensemble
├── Dataset/
│   ├── Fake.csv.zip                          # Fake news articles (Kaggle dataset)
│   └── True.csv.zip                          # Real news articles (Kaggle dataset)
└── Research Paper/
    └── Fake_News_Detection_Research_Paper.pdf # Full academic write-up of methodology & results
```

## 🚀 Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/Pari219/Fake-News-Detection.git
   ```
2. Unzip the dataset files inside `Dataset/` and place `Fake.csv` and `True.csv` in your Google Drive (or update the file paths in the notebook to point to a local directory)
3. Open `Fake_News_Detection_GloVe_WordNet.ipynb` in Google Colab
4. Run all cells — the notebook will:
   - Download and load GloVe embeddings automatically
   - Preprocess and lemmatize the text
   - Train all six classifiers
   - Build the hybrid ensemble
   - Generate all EDA visualizations, confusion matrices, and performance comparison plots

## 📄 Research Paper

The complete methodology, related work, mathematical formulation of each classifier, and detailed results analysis are documented in [`Fake_News_Detection_Research_Paper.pdf`](./Research%20Paper/Fake_News_Detection_Research_Paper.pdf), authored by **Karishma Rajput, Kamran Ahmed, and Ajisth Shukla** — Computer Science & Engineering (Data Science), NSUT.

## 🔮 Future Work

- Explore transformer-based architectures (BERT, RoBERTa) to better capture contextual semantics beyond averaged embeddings
- Integrate explainable AI techniques (LIME, SHAP) for per-prediction interpretability
- Incorporate social propagation and metadata features for improved real-world generalization
- Extend to real-time deployment and cross-domain testing

## 👩‍💻 Author

**Karishma Rajput (Pari)**
B.Tech, Computer Science & Engineering (Data Science), NSUT Delhi
[GitHub: @Pari219](https://github.com/Pari219)

*Co-authored with Kamran Ahmed and Ajisth Shukla.*

---

⭐ If you find this project interesting, consider giving it a star!

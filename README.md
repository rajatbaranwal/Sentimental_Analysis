# Sentiment Analysis using VADER and RoBERTa

This project performs sentiment analysis on a subset of Amazon food reviews using two powerful models: **VADER** (Valence Aware Dictionary and sEntiment Reasoner) and **RoBERTa** (a transformer-based model from Hugging Face).

## 📊 Dataset
We use the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset. For faster experimentation, only the first 500 reviews are processed.

## 🔧 Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- NLTK
- Hugging Face Transformers
- Pretrained Models: `cardiffnlp/twitter-roberta-base-sentiment`

## 🧠 Workflow Overview

### 1. Exploratory Data Analysis (EDA)
- Bar plots to show the distribution of review scores (1 to 5 stars).

### 2. VADER Sentiment Analysis
- Basic text preprocessing and tokenization using NLTK.
- Applied VADER sentiment scoring (`neg`, `neu`, `pos`, `compound`) to each review.

### 3. RoBERTa Sentiment Analysis
- Used pretrained RoBERTa model to classify reviews into `Negative`, `Neutral`, and `Positive`.
- Applied softmax to convert logits to probabilities.

### 4. Combined Analysis
- Compared VADER and RoBERTa results.
- Visualized sentiment distributions and correlation with review scores.

### 5. Random Example Analysis
- Tested sentiment scoring on a random review from the dataset.
- Also allows testing on custom text inputs.

## 📈 Visualizations
- Compound sentiment scores vs review stars.
- Sentiment probability plots per review score.
- Final sentiment distribution (Positive, Negative, Neutral).

## 📌 Key Results
- Sentiment classification from both rule-based (VADER) and context-aware (RoBERTa) models.
- RoBERTa showed stronger nuance in complex sentences, while VADER was faster.

## ▶️ How to Run
1. Clone the repo.
2. Open `Sentimental_Analysis.ipynb` in [Google Colab](https://colab.research.google.com) or locally in Jupyter.
3. Make sure to install dependencies (`nltk`, `transformers`, `tqdm`).
4. Run the notebook cells sequentially.

## 📚 License
This project is for educational and research purposes.

---

Feel free to update this README based on any enhancements or model changes you introduce!

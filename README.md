# Predictive Task: Spoiler Detection in Movie Reviews

## 1. Task Definition

**Predictive Task:** Binary classification to predict whether a movie review contains spoilers (`is_spoiler = True/False`).

**Input:**
- Review text
- Movie metadata (synopsis, summary, genre, rating, release date)
- User metadata (user ID, historical behavior)

**Output:**
- Binary prediction: Spoiler (1) or Not Spoiler (0)

---

## 2. Evaluation Metrics

| Metric | Why Use It |
|--------|------------|
| **Accuracy** | Overall correctness; easy to interpret |
| **F1-Score** | Balances precision/recall; important for imbalanced classes (26% spoilers vs 74% non-spoilers) |
| **Precision** | "Of reviews flagged as spoilers, how many actually are?" (minimize false positives) |
| **Recall** | "Of actual spoilers, how many did we catch?" (minimize false negatives) |

**Primary Metric:** F1-Score (due to class imbalance)

**Data Split:** 90% Train / 5% Validation / 5% Test (stratified by label)

---

## 3. Baselines and Model Progression

| Model | Description |
|-------|-------------|
| **1. Bag of Words + Logistic Regression （Baseline)** | Simple word frequency counts as features |
| **2. TF-IDF + Logistic Regression** | Word importance weighting with inverse document frequency |
| **3. TF-IDF + Synopsis/Summary + Logistic Regression** | Add movie plot text (synopsis or summary as fallback) |
| **4. TF-IDF + Synopsis/Summary + Cosine Similarity + User Features + Movie Features + Logistic Regression** | Add review-plot similarity, user spoiler history, and movie spoiler rate |
| **5. LSTM + TF-IDF + Synopsis/Summary + Cosine Similarity + User Features + Movie Features** | Deep learning model combining sequential text understanding with all engineered features |

**Progression Logic:**
1. **Baseline:** Start with simple text features (BOW)
2. **Improvement:** Better text representation (TF-IDF)
3. **Contextual:** Compare review against known plot information
4. **Metadata:** Incorporate user behavior patterns and movie characteristics
5. **Deep Learning:** Capture sequential patterns and word relationships with LSTM

---

## 4. Validity Assessment

### A. Data Reliability
- **Source:** IMDB Spoiler Dataset from Kaggle ([link](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset)) by Rishabh Misra, updated 7 years ago. It was collected from IMDB. It was collected for the achieving the same goal as ours: to identify these spoilers in entertainment reviews, so that users can more effectively navigate review platforms.
- **Size:** 573,913 reviews across 1,572 movies
- **Labels:** Ground truth `is_spoiler` labels provided by IMDB users who self-reported spoilers
- **Quality Checks:** 
  - No missing review text
  - All movie IDs successfully mapped to movie details
  - Data cleaned for ID inconsistencies (trailing slashes)

### B. Model Comparison
- Multiple models trained and compared on the same train/val/test split
- Progression from simple baselines (BOW) to complex models (LSTM + all features)
- Each model evaluated on identical held-out test set for fair comparison
- Results summarized in comparison table with rankings

### C. Evaluation Metrics
- **Accuracy:** Overall percentage of correct predictions
- **F1-Score:** Harmonic mean of precision and recall (important for imbalanced data)
- **Classification Report:** Precision, recall, and F1 for both classes
- All metrics computed on held-out test set (never seen during training)

### D. Manual Validation (Sanity Check)
Test the model with manually crafted examples:

**Likely Spoiler:**
> "I can't believe Bruce Willis was dead the entire time! The twist at the end where we find out he was a ghost all along completely shocked me."

**Likely Non-Spoiler:**
> "Great movie with amazing cinematography. The acting was superb and I highly recommend it to anyone who enjoys thrillers."

**Expected Behavior:**
- Model should predict **Spoiler** for the first example (contains plot twist revelation)
- Model should predict **Non-Spoiler** for the second example (general review without plot details)

---

# Data Preprocess


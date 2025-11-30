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

## * Rerunning the file may take around 6 hours

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
| **6. S-Bert + Cosine Similarity + Logistic Regression** | Highly complicated neural network that captures the semantic relationship between the review and the plot synopsis by calculating the cosine similarity of their S-Bert embeddings |

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

# Data Preprocess / Cleaning

#### Movie ID Conflict (`tt0104014` vs. `tt0104014/`)

The data strongly indicates that the movie IDs `tt0104014` (Reviews) and `tt0104014/` (Details) refer to the **exact same movie**. The trailing slash (`/`) in the details record is a data artifact that prevents a clean join.


#### 1. Review Record (`tt0104014`)

The review text provides the film's title directly:

```json
{
  "review_date": "16 July 2012",
  "movie_id": "tt0104014",
  "user_id": "ur5358902",
  "is_spoiler": false,
  "review_text": "**Tinto Brass** is usually referred to as either a misunderstood genius or a talentless hack. **Cosi fan tutte (\"All Ladies Do It\")** proves that he's neither one. When he doesn't take himself too seriously (which, unfortunately, he does quite often) the man is perfectly capable of creating fun, well-polished, strict entertainment. **Cosi fan tutte** certainly slips into the realm of pornography on occasion, but it has a sense of lightness and fun that breaks the ice and avoids real discomfort...",
  "rating": "7",
  "review_summary": "Silly and Sexy"
}
```

#### 2. Detail Record (`tt0104014/`)

The detail record provides a plot summary that is perfectly consistent with a **Tinto Brass** film titled **`Così fan tutte`**:

```json
{
  "plot_summary": "For a while now, beautiful 24-year-old **Diana Bruni** who's been happily married for five years, has been feeling distressed, experiencing an inexplicable, rather restless craving to finally live her life to the full and to break free from what society forbids. As this urge grows stronger by the day, Diana will ultimately yield to her **carnal longings**, and through a parade of particularly explicit nocturnal **sensual adventures**, she will utterly embrace passion, even if this comes by way of **transgression**. However, before long, the unaware husband, **Paolo**, will find all about his headstrong and disobedient wife's **extra-marital escapades**...",
  "movie_id": "tt0104014/",
  "duration": "1h 33min",
  "genre": ["Comedy", "Drama"],
  "rating": "5.3",
  "release_date": "1992-02-21",
  "plot_synopsis": ""
}
```
Conclusion: They are indeed the same movie and thus hsould have the same movie_id.

# Creating features

#### User feature

1. Create a dataset called user_df 

| Feature | Description | Example |
|---------|-------------|---------|
| **user_id** | Unique identifier for each user | `ur0123456` |
| **review_count** | Total number of reviews written by this user | `15` |
| **avg_rating** | Average rating the user gives to movies (1-10 scale) | `7.2` |
| **user_smoothed_spoiler_rate** | Bayesian-smoothed probability that this user writes spoilers. (use review number, spoil rate to determine) | `0.42` (42% spoiler rate) |
| **user_confidence** | How much we can trust the user's spoiler rate based on their review count. Higher = more reliable. | `0.75` (75% confident) |
| **user_type** | Categorical label classifying user behavior | `always_spoils`, `never_spoils`, `usually_spoils`, `rarely_spoils`, `mixed`, `uncertain` |

---

#### Movie Feature

1. add column to the end of reviews_df
    - num_reviews
    - Spoiler_percent (per movie)

# Models

### Bag of Words + Logistic Regression

**Advantages**: 
 - Extremely fast training and prediction
 - Very space-efficient feature matrix (sparse)
 - Easy to deploy, we only need the review text of the movie as input

**Disadvantages**:
 - Low accuracy, if we used `class_weight='balanced'`, the accuracy would be even lower
 - Semantic Blindness: Treats "good" and "not good" as completely different words; loses word order/context

### TFIDF + Logistic Regression

1. TF-IDF for review text + synopsis to do classification
2. TF-IDF cosine compare the review text with plot_synopsis serve as feature
3. TF-IDF cosine + review text + synopsis + user features + movie features

**Advantages**: 
 - Still very fast training and prediction
 - More features available for the model
 - With the correct features, can achieve very high accuracy, precision, and recall

**Disadvantages**:
 - Relies heavily on feature engineering
 - Still ignores sequence and context
 - Can lead to high memory/computation during matrix combination

### LSTM + TF-IDF + Cosine + User/Movie Features

**Advantages**: 
 - Captures long-range dependencies and word order (sequential information) essential for narrative spoilers
 - Strongest performance ceiling by combining sequential text features with structured meta-features.
 
**Disadvantages**:
 - Extremely slow to train
 - Requires high computational resources (GPU)
 - Difficult to debug and less interpretable
 - High memory usage for storing embeddings and model weights

### S-Bert + Cosine Similarity + Logistic Regression

**Advantages**: 
 - Produces high-quality, dense, contextual embeddings (sentence-level meaning)
 - Benefits from pre-training on massive text corpora
 - Fast prediction once the embeddings are generated
 
**Disadvantages**:
 - Extremely slow to train
 - High memory usage for storing embeddings and model weights
 - Reduces a whole review and synopsis into a single vector, losing fine-grained word information
 - Both S-BERT and BERT relies heavily on fine-tuning, but due to the sheer size of this S-BERT model and the excessively long execution time per run, in-depth adjustments are not very possible

# Evaluation

Metric Definitions

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Accuracy** | (TP + TN) / Total | Overall % of correct predictions |
| **Precision** | TP / (TP + FP) | "Of all predicted spoilers, how many actually were spoilers?" |
| **Recall** | TP / (TP + FN) | "Of all actual spoilers, how many did we catch?" |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision & recall (balances both) |
| **Support** | — | Number of actual samples in each class |

Where:
- **TP** = True Positives (correctly predicted spoilers)
- **TN** = True Negatives (correctly predicted non-spoilers)
- **FP** = False Positives (predicted spoiler, but wasn't)
- **FN** = False Negatives (missed spoiler, predicted non-spoiler)

---

Results Analysis

| Model | Accuracy | F1 (Spoiler) | Precision | Recall |
|-------|----------|--------------|-----------|--------|
| BOW | 77.2% | 0.46 | 0.61 | 0.37 |
| TF-IDF + Synopsis | 78.8% | 0.48 | 0.67 | 0.37 |
| TF-IDF + Cosine | 78.7% | 0.48 | 0.67 | 0.37 |
| **TF-IDF + User + Movie** | **91.3%** | **0.82** | **0.87** | **0.78** |
| **Hybrid LSTM** | **92.5%** | **0.87** | **0.82** | **0.93** |
| S-Bert | 73.69% | 0 | 0 | 0 |

---

Why Hybrid LSTM is Best

1. Highest Accuracy (92.5%)
- Correctly classifies 93% of all reviews

2. Highest F1-Score (0.87)
- Best balance between precision and recall for spoiler detection

3. Highest Recall (0.93)
- **Catches 93% of actual spoilers** (vs only 37% for basic TF-IDF)
- Critical for spoiler detection: missing a spoiler is worse than false alarm

4. Strong Precision (0.82)
- When it says "spoiler," it's right 82% of the time

---

Key Insights

Why Basic Models (BOW, TF-IDF) Struggled:
- High accuracy (~78%) is **misleading** — mostly predicting "non-spoiler"
- Recall of 0.37 = **Only catching 37% of spoilers** (missing 63%!)
- The model takes the "easy path" — just predict majority class

Why User + Movie Features Helped (+13% accuracy):
- **User history matters**: Users who always spoil will likely spoil again
- **Movie patterns matter**: Some movies get spoiled more than others
- These features provide **behavioral priors** before even reading the text

Why LSTM Added More (+1% accuracy, +5% F1):
- Captures **sequential patterns** in language
- Understands context: "He dies" after setup vs "He dies" in isolation
- Attention mechanism focuses on spoiler-indicating phrases

---

## Related Work and Things We Can Improve On

Another project that used this dataset used Naive Bayes Multinomial and Support Vector Machine for the same task. Both models ultimately achieved around 80% accuracy, which a lower score than ours. Yet that does not mean that the project cannot shine new light upon us. The project considered new features revoling around the rating of a review. It displayed relationships between how likely it is for a user to write a review with spoiler according to the rating they gave. It also compared the time the movie was out to the time when the review was written. These are features that we did not include in our models but they do seem very interpretable. Unfortunately, the author of this project did not display too many final statistics for his project, so we are unable to tell if these features do a significantly better job than ours.

We can introduce advanced linguistic or structural features, particularly for the LogReg models. It is logical to think that spoilers often appear near the end of a review, are short, or involve named entities (characters/locations). Hence advance features like this may further increase our accuracy.

We may want to re-evaluate the scaling of non-text features (User, Movie, Cosine Sim) before concatenation with the sparse TF-IDF matrix.

We can try replacing the standard LSTM in the hybrid model with a Bi-directional LSTM (Bi-LSTM) or a Gated Recurrent Unit (GRU), which captures context from both past and future words. This can be highly beneficial for sequence tasks like spotting spoilers (the crucial context might appear after the spoiler itself).

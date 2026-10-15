# ANN Spam Message Detector

A simple Artificial Neural Network (ANN) that classifies text messages as **Spam** or **Ham** (not spam). Messages are converted to TF-IDF vectors and passed to a small Keras model. A Streamlit app lets you test messages in the browser.

## Project Structure

| File | Description |
|------|-------------|
| `spam.csv` | Labeled dataset (`label`, `message`): 100 messages, 50 ham / 50 spam |
| `experiment.ipynb` | Cleans the data, trains the ANN, evaluates it and saves the model |
| `app.py` | Streamlit web app for predictions |
| `ann_spam_model.keras` | Trained model (created by the notebook) |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (created by the notebook) |
| `requirements.txt` | Python dependencies |

## How It Works

1. **Preprocessing:** labels are normalized, empty rows and duplicates are removed, and `ham`/`spam` is encoded as `0`/`1`.
2. **Split:** 75% training and 25% testing, stratified by label.
3. **Vectorization:** `TfidfVectorizer` with unigrams and bigrams (up to 3000 features), fitted on the training data only.
4. **Model:**
   ```
   Input (TF-IDF features)
   → Dense(64, ReLU) → Dropout(0.2)
   → Dense(64, ReLU) → Dropout(0.2)
   → Dense(1, Sigmoid)
   ```
   Trained with the Adam optimizer and binary cross-entropy loss: batch size 8, up to 100 epochs, and early stopping (patience 10, best weights restored).
5. **Prediction:** a probability of 0.5 or higher is classified as **Spam**.

## Results

On the 25-message test set:

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Ham   | 0.92 | 0.92 | 0.92 |
| Spam  | 0.92 | 0.92 | 0.92 |
| **Accuracy** | | | **0.92** |

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

**1. Train the model.** Open `experiment.ipynb` and run all cells. This creates `ann_spam_model.keras` and `tfidf_vectorizer.pkl`.

**2. Run the app:**

```bash
streamlit run app.py
```

Enter a message and click **Check Message** to see the spam probability and prediction.

| Example message | Spam probability | Prediction |
|-----------------|------------------|------------|
| Hey, are we still meeting for lunch tomorrow? | 0.069 | Not spam |
| Congratulations! Claim your free prize now | 0.998 | Spam |

## Limitations

The dataset is very small (100 messages), so the model may misclassify messages that don't look like its training examples. For better accuracy, train on a larger dataset such as the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (about 5,500 messages).

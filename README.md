# ANN Implementation

Three Artificial Neural Network projects built with TensorFlow/Keras, each with a Streamlit app for predictions. All three share one virtual environment and one `requirements.txt` at the root.

## Projects

| Project | Task | Dataset | Model file |
|---|---|---|---|
| [ANN-Classification](ANN-Classification/README.md) | Predict whether a bank customer will leave the bank (churn) | `Churn_Modelling.csv` (10,000 customers) | `ann_model.h5` |
| [ANN-Regression](ANN-Regression/) | Predict a bank customer's estimated salary | `Churn_Modelling.csv` | `ann_regression_model.h5` |
| [ANN-Classification-Spam-Detection](ANN-Classification-Spam-Detection/README.md) | Classify a text message as spam or ham, using TF-IDF features | `spam.csv` (100 messages) | `ann_spam_model.keras` |

See each project's README for details on its data, model and results.

## Structure

```
ANN-Implementation/
├── requirements.txt                    # shared dependencies
├── venv/                               # shared virtual environment (not committed)
├── ANN-Classification/
├── ANN-Regression/
└── ANN-Classification-Spam-Detection/
```

## Setup

Requires **Python 3.11**.

```powershell
cd ANN-Implementation
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/Scripts/activate       # Windows Git Bash
# source venv/bin/activate           # macOS / Linux

pip install -r requirements.txt
```

In VS Code, select `venv\Scripts\python.exe` as the interpreter and the notebook kernel.

## Run an app

With the venv active, run each app from inside its own folder, because the apps load their model and `.pkl` files from relative paths:

```powershell
cd ANN-Classification;                streamlit run app.py
cd ANN-Regression;                    streamlit run app.py
cd ANN-Classification-Spam-Detection; streamlit run app.py
```

The app opens at http://localhost:8501.

## Retrain a model

Open the project's training notebook and run all cells. This regenerates the model file and the fitted encoders, scaler or vectorizer:

| Project | Notebook |
|---|---|
| ANN-Classification | `experiments.ipynb` (and `prediction.ipynb` for a single-example walkthrough) |
| ANN-Regression | `experiment.ipynb` |
| ANN-Classification-Spam-Detection | `experiment.ipynb` |

To view training curves for the churn and regression models in TensorBoard:

```powershell
tensorboard --logdir ANN-Classification/logs/fit
tensorboard --logdir ANN-Regression/logs/fit
```

## Notes

- The `.h5` models were saved with an older Keras 2. They load in the current Keras 3, but `ANN-Classification/app.py` loads its model with `compile=False` because Keras 3 can't restore that model's saved loss function. Predictions are unaffected.
- `venv/`, `logs/`, `.ipynb_checkpoints/`, `__pycache__/` and `.vscode/` are excluded by the root `.gitignore`.

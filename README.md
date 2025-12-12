# Stochastic Weight Averaging (SWA) for Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)
![NLP](https://img.shields.io/badge/AI-SWA_Optimization-green)

## 📄 Project Overview
This project implements **Stochastic Weight Averaging (SWA)** to improve the generalization of Deep Learning models on the **MELD Dataset** (Multimodal EmotionLines Dataset). 

While many models suffer from overfitting on small conversational datasets, this project demonstrates how averaging weights along the trajectory of SGD (Stochastic Gradient Descent) leads to flatter minima and better test performance.

## 📂 Repository Contents

| File | Description |
| :--- | :--- |
| **`SWA_code_final.ipynb`** | The complete PyTorch pipeline: Custom `SWA` optimizer implementation, RNN/LSTM architecture, and training loops using `torchtext`. |
| **`Report.pdf`** | Technical report detailing the experiment results and the impact of SWA on accuracy. |

## 🛠️ Tech Stack & Methodology

### 1. Text Preprocessing
* **Tokenization:** utilized `spaCy` for efficient tokenization.
* **Embeddings:** Pre-trained **GloVe (6B, 100d)** vectors were mapped to the vocabulary to capture semantic meaning.
* **Data Loading:** Handled via `torchtext.legacy.data` for batching and padding.

### 2. Model Architecture (PyTorch)
The model is a recurrent neural network built for sequence classification:
* **Embedding Layer:** Frozen GloVe weights.
* **Encoder:** Bidirectional **LSTM** (Long Short-Term Memory) to capture context from both directions of the conversation.
* **Optimization:** **Stochastic Weight Averaging (SWA)**. Instead of using the final model weights, the model averages weights at different stages of training to find a more robust solution.

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Shourya0902/swa-text-emotion-recognition.git](https://github.com/Shourya0902/swa-text-emotion-recognition.git)
Install Dependencies

Bash

pip install -r requirements.txt
python -m spacy download en_core_web_sm
Data Setup Ensure train_sent_emo.csv and test_sent_emo.csv (from MELD) are in the root directory.

Run Open SWA_code_final.ipynb to execute the training.

👨‍💻 Author
Shourya Marwaha MSc Data Science & Analytics | MBA

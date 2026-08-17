# 🔍 Text Classification & Sentiment Analysis System

An NLP-based text classification web app that automatically detects and categorizes social media content into Hate Speech, Offensive Speech, or Neutral using machine learning. Built with Python and Streamlit.

## 📌 Features
- Classifies text into 3 categories: Hate Speech, Offensive Speech, No Hate/Offensive Speech
- Full NLP preprocessing pipeline: tokenization, stop-word removal, stemming
- TF-IDF vectorization for converting text to numerical features
- Naive Bayes classification with accuracy and full classification report displayed
- Real-time text inference via interactive Streamlit UI

## 🛠️ Tech Stack
| Layer | Tools |
|---|---|
| Language | Python |
| NLP | NLTK, Scikit-learn |
| Vectorization | TF-IDF (TfidfVectorizer) |
| Model | Multinomial Naive Bayes |
| Evaluation | Accuracy Score, Precision, Recall, F1-Score |
| Frontend | Streamlit |

## 🧠 How It Works

1. **Data Loading** — Twitter dataset (24,000+ records) with labeled hate/offensive/neutral tweets
2. **Preprocessing** — Lowercasing, URL removal, punctuation stripping, stop-word removal
3. **Vectorization** — TF-IDF converts cleaned text into numerical feature vectors
4. **Model Training** — Multinomial Naive Bayes trained on an 80/20 train-test split
5. **Evaluation** — Accuracy score and full classification report (precision, recall, F1) computed
6. **Inference** — User inputs any text and receives an instant prediction

## 📁 Project Structure
Hate_Speech_Detection/
│
├── app.py # Streamlit app + full ML pipeline
├── twitter.csv # Labeled Twitter dataset (24,000+ records)
├── requirements.txt # Python dependencies
├── Procfile # Deployment configuration
└── .gitignore


## ⚙️ Installation & Setup
# 1. Clone the repository
git clone https://github.com/Krish-1710/Hate_Speech_Detection.git
cd Hate_Speech_Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

## 📊 Dataset
- Source: Twitter hate speech dataset
- Size: 24,000+ labeled tweets
- Classes: Hate Speech (0), Offensive Speech (1), No Hate/Offensive Speech (2)

## 🔮 Future Improvements
- Compare additional models (SVM, Logistic Regression, BERT)
- Add confidence scores to predictions
- Extend to multi-language support

## 👤 Author
**Krishkumar Patel**
- GitHub: [@Krish-1710](https://github.com/Krish-1710)
- LinkedIn: [krish-patel-213bab2a6](https://www.linkedin.com/in/krish-patel-213bab2a6/)

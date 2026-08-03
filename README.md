<h2 align="center">Tolga Şakar</h2>
<p align="center"><b>AI & ML Engineer · NLP Researcher</b></p>

<p align="center">
  <a href="https://orcid.org/0009-0009-3684-9755"><img src="https://img.shields.io/badge/ORCID-A6CE39?style=flat-square&logo=orcid&logoColor=white" alt="ORCID"></a>
  <a href="https://www.linkedin.com/in/tolga-sakar/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="https://huggingface.co/lonewolflab"><img src="https://img.shields.io/badge/HuggingFace-lonewolflab-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://www.kaggle.com/dfavenfre"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle"></a>
  <a href="https://arxiv.org/a/sakar_t_1"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="mailto:tolgasa2@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>
</p>

---

Production-grade multimodal AI systems, autonomous agents ve NLP pipeline'ları üzerine çalışıyorum.

Bağımsız araştırmalarım **morphologically-aware neural tokenization**, **word representations** ve düşük kaynaklı / eklemeli diller için **Retrieval-Augmented Generation** üzerine yoğunlaşıyor — [`lonewolf-rd`](https://github.com/lonewolf-rd) çatısı altında.

---

## 📰 Publications

### Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish
*arXiv preprint, 2026 — sole author*

Türkçe için **kayıpsız (lossless), morfoloji-farkında bir neural tokenizer *ve* word embedder**. Türevlenebilir Poisson–binomial soft segmentation ile yüzey-koruyan, birebir morfem bölütlemesi üretiyor (`decode(encode(w)) = w`); aynı forward pass yapısal word embedding'leri de veriyor.

- Tersinir tokenizer'lar arasında **en düşük BPC**
- BPE / WordPiece / Unigram'a kıyasla **~2× morfolojik hizalama**
- Lexical retrieval'da BERTurk ve BGE-M3'ün önünde

<a href="https://github.com/lonewolf-rd/TurkishMorpheus"><img src="https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github&logoColor=white"></a>
<a href="https://huggingface.co/lonewolflab/Morpheus-TR-50K"><img src="https://img.shields.io/badge/Model-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
<a href="https://huggingface.co/spaces/lonewolflab/morpheus-tr-demo"><img src="https://img.shields.io/badge/Demo-FF6F00?style=flat-square&logo=gradio&logoColor=white"></a>
<a href="https://arxiv.org/abs/2606.18717"><img src="https://img.shields.io/badge/Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white"></a>

### Maximizing RAG Efficiency: A Comparative Analysis of RAG Methods
*Natural Language Processing, Cambridge University Press (SCI Q1), 2025*

Vector store, embedding modeli ve LLM kombinasyonları üzerinde **23.625 konfigürasyonluk** bir grid-search çalışması; retrieval kalitesi, benzerlik temelli sıralama, token tüketimi, çalışma süresi ve donanım kullanımı arasındaki ödünleşimleri cross-domain veriyle ölçüyor. **Contextual compression filtreleri** token tüketimini ve donanım yükünü belirgin şekilde düşürüyor — benzerlik tarafındaki kayıp ise RAG yöntemine ve kullanım senaryosuna göre çoğu zaman kabul edilebilir seviyede kalıyor.

<a href="https://www.cambridge.org/core/journals/natural-language-processing/article/maximizing-rag-efficiency-a-comparative-analysis-of-rag-methods/D7B259BCD35586E04358DF06006E0A85"><img src="https://img.shields.io/badge/Paper-1B4F72?style=flat-square&logo=cambridge&logoColor=white"></a>
<a href="https://github.com/dfavenfre/dfavenfre/blob/main/maximizing-rag-efficiency-a-comparative-analysis-of-rag-methods.pdf"><img src="https://img.shields.io/badge/PDF-EC1C24?style=flat-square&logo=adobeacrobatreader&logoColor=white"></a>

---

## 🖥️ Open-Source Projects

### 🤖 LLM Systems & Agents

| Project | Tech Stack |
|---|---|
| [**Multi-Modal RAG**](https://github.com/dfavenfre/MultiModal-RAG) | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square&logo=chromadb&logoColor=white) |
| [**RAG Optimization**](https://github.com/dfavenfre/RAG-Optimization) | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?style=flat-square&logo=langsmith&logoColor=white) ![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white) |
| [**TalkYou**](https://github.com/dfavenfre/TalkYou) | ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langgraph&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| [**LLMRoboFund**](https://github.com/dfavenfre/LLMRoboFund) | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square&logo=chromadb&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |

### 🧠 Deep Learning & Computer Vision

| Project | Tech Stack |
|---|---|
| [**Olivetti Face Recognition**](https://github.com/dfavenfre/Olivetti-Faces-PyTorch) | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| [**MobileNetV1 — Julia**](https://github.com/dfavenfre/MobileNet-Julia) | ![Julia](https://img.shields.io/badge/Julia-9558B2?style=flat-square&logo=julia&logoColor=white) ![Flux](https://img.shields.io/badge/Flux-9558B2?style=flat-square&logo=julia&logoColor=white) ![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black) |
| [**EfficientNetV2 Transfer Learning**](https://github.com/dfavenfre/Transfer-Learning-CNN-Fine-Tuning) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black) |
| [**Food Vision**](https://github.com/dfavenfre/Food-Vision-Tensorflow) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black) |
| [**Fashion MNIST**](https://github.com/dfavenfre/Fashion-MNIST-Tensorflow) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black) |
| [**Financial Sentiment Classifier**](https://github.com/dfavenfre/financial-sentiment-classifier) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) |

### 📈 Forecasting & Tabular ML

| Project | Tech Stack |
|---|---|
| [**Electricity Price Forecasting**](https://github.com/dfavenfre/electricity-price-forecasting) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square) |
| [**Bitcoin Price Forecasting**](https://github.com/dfavenfre/Bitcoin-Price-Forecasting) | ![pmdarima](https://img.shields.io/badge/pmdarima-3776AB?style=flat-square&logo=python&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) |
| [**Bike Sharing Demand**](https://github.com/dfavenfre/Bike-Sharing-Demand-Prediction) | ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square) ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square) ![Optuna](https://img.shields.io/badge/Optuna-2B6CB0?style=flat-square&logo=optuna&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| [**Bank Deposit Prediction**](https://github.com/dfavenfre/customer_deposit_classifier) | ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| [**Credit Score Prediction**](https://github.com/dfavenfre/Credit-Score-Prediction) | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| [**Econ Dashboard**](https://github.com/dfavenfre/Econ-Dashboard) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |

---

## 📊 GitHub Stats

<p align="center">
  <img alt="Profile Details" src="https://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=dfavenfre&theme=dracula" />
</p>

<p align="center">
  <img height="200em" alt="Repos per Language" src="https://github-profile-summary-cards.vercel.app/api/cards/repos-per-language?username=dfavenfre&theme=dracula" />
  <img height="200em" alt="Most Commit Language" src="https://github-profile-summary-cards.vercel.app/api/cards/most-commit-language?username=dfavenfre&theme=dracula" />
</p>

<p align="center">
  <img height="200em" alt="Stats" src="https://github-profile-summary-cards.vercel.app/api/cards/stats?username=dfavenfre&theme=dracula" />
  <img height="200em" alt="Productive Time" src="https://github-profile-summary-cards.vercel.app/api/cards/productive-time?username=dfavenfre&theme=dracula&utcOffset=3" />
</p>

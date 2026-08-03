<h2 align="center">Tolga Şakar</h2>
<p align="center"><b>AI & ML Engineer · NLP Researcher</b></p>

<p align="center">
  <a href="https://orcid.org/0009-0009-3684-9755"><img src="https://img.shields.io/badge/ORCID-A6CE39?style=flat-square&logo=orcid&logoColor=white" alt="ORCID"></a>
  <a href="https://huggingface.co/lonewolflab"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://www.linkedin.com/in/tolga-sakar/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="mailto:tolgasa2@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>
</p>

<p align="center">
  Building production-grade multimodal AI systems, autonomous agents, and NLP pipelines.<br>
  Independent research on tokenization, word representations, and retrieval<br>
  for low-resource and agglutinative languages.
</p>

---
## Publications

**Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish**
*arXiv preprint, 2026 — sole author*

A lossless, morphology-aware neural tokenizer *and* word embedder for Turkish. A differentiable Poisson–binomial soft segmentation produces exact, surface-preserving morpheme splits (`decode(encode(w)) = w`), while the same forward pass yields structured word embeddings. Achieves the lowest BPC among reversible tokenizers and roughly 2× the morphological alignment of BPE, WordPiece and Unigram, and leads BERTurk and BGE-M3 on lexical retrieval.

[Paper](https://arxiv.org/abs/2606.18717) · [Repository](https://github.com/lonewolf-rd/TurkishMorpheus) · [Model](https://huggingface.co/lonewolflab/Morpheus-TR-50K) · [Demo](https://huggingface.co/spaces/lonewolflab/morpheus-tr-demo)

**Maximizing RAG Efficiency: A Comparative Analysis of RAG Methods**
*Natural Language Processing, Cambridge University Press (SCI Q1), 2025*

A grid-search study of 23,625 configurations across vector stores, embedding models and LLMs on cross-domain data, quantifying the trade-offs between retrieval quality, similarity-based ranking, token usage, runtime and hardware utilization. Contextual compression filters substantially reduce token consumption and hardware load, at a similarity cost that is often acceptable depending on the RAG method and use case.

[Paper](https://www.cambridge.org/core/journals/natural-language-processing/article/maximizing-rag-efficiency-a-comparative-analysis-of-rag-methods/D7B259BCD35586E04358DF06006E0A85) · [PDF](https://github.com/dfavenfre/dfavenfre/blob/main/maximizing-rag-efficiency-a-comparative-analysis-of-rag-methods.pdf)

## Research

Ongoing work lives under [**lonewolf-rd**](https://github.com/lonewolf-rd). Released models and demos are on [**lonewolflab**](https://huggingface.co/lonewolflab).

| Project | Description |
|---|---|
| [TurkishMorpheus](https://github.com/lonewolf-rd/TurkishMorpheus) | Lossless, morphology-aware neural tokenizer and word embedder for Turkish |

## Projects

### LLM Systems & Agents

| Project | Tech Stack |
|---|---|
| [**RAG Optimization**](https://github.com/dfavenfre/RAG-Optimization) | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?style=flat-square&logo=langsmith&logoColor=white) ![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white) |
| [**Multi-Modal RAG**](https://github.com/dfavenfre/MultiModal-RAG) | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square&logo=chromadb&logoColor=white) |
| [**TalkYou**](https://github.com/dfavenfre/TalkYou) | ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langgraph&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| [**LLMRoboFund**](https://github.com/dfavenfre/LLMRoboFund) | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square&logo=chromadb&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |

### Deep Learning & Computer Vision

| Project | Tech Stack |
|---|---|
| [**Olivetti Face Recognition**](https://github.com/dfavenfre/Olivetti-Faces-PyTorch) | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| [**MobileNetV1 — Julia**](https://github.com/dfavenfre/MobileNet-Julia) | ![Julia](https://img.shields.io/badge/Julia-9558B2?style=flat-square&logo=julia&logoColor=white) ![Flux](https://img.shields.io/badge/Flux-9558B2?style=flat-square&logo=julia&logoColor=white) ![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black) |
| [**EfficientNetV2 Transfer Learning**](https://github.com/dfavenfre/Transfer-Learning-CNN-Fine-Tuning) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black) |
| [**Food Vision**](https://github.com/dfavenfre/Food-Vision-Tensorflow) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black) |
| [**Fashion MNIST**](https://github.com/dfavenfre/Fashion-MNIST-Tensorflow) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![W&B](https://img.shields.io/badge/W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black) |
| [**Financial Sentiment Classifier**](https://github.com/dfavenfre/financial-sentiment-classifier) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) |

### Forecasting & Tabular ML

| Project | Tech Stack |
|---|---|
| [**Electricity Price Forecasting**](https://github.com/dfavenfre/electricity-price-forecasting) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square) |
| [**Bitcoin Price Forecasting**](https://github.com/dfavenfre/Bitcoin-Price-Forecasting) | ![pmdarima](https://img.shields.io/badge/pmdarima-3776AB?style=flat-square&logo=python&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) |
| [**Bike Sharing Demand**](https://github.com/dfavenfre/Bike-Sharing-Demand-Prediction) | ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square) ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square) ![Optuna](https://img.shields.io/badge/Optuna-2B6CB0?style=flat-square&logo=optuna&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| [**Bank Deposit Prediction**](https://github.com/dfavenfre/customer_deposit_classifier) | ![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| [**Credit Score Prediction**](https://github.com/dfavenfre/Credit-Score-Prediction) | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| [**Econ Dashboard**](https://github.com/dfavenfre/Econ-Dashboard) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |

## Metrics

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./metrics-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./metrics-light.svg">
    <img alt="GitHub metrics" src="./metrics-light.svg">
  </picture>
</p>

## Contact

Open to research collaboration on tokenization, representation learning and retrieval for morphologically rich languages — [tolgasa2@gmail.com](mailto:tolgasa2@gmail.com)
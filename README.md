<h3 align="center">Tolga Şakar — AI & ML Engineer · NLP Researcher</h3>

<p align="center">
<br/>
<a href="https://orcid.org/0009-0009-3684-9755">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=flat-square&logo=orcid&logoColor=white" alt="ORCID">
</a>
<a href="https://www.linkedin.com/in/tolga-sakar/">
    <img src="https://img.shields.io/badge/-Linkedin-blue?style=flat-square&logo=linkedin">
</a>
<a href="https://huggingface.co/lonewolflab">
    <img src="https://img.shields.io/badge/HuggingFace-lonewolflab-yellow?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace">
</a>
<a href="https://www.kaggle.com/dfavenfre">
      <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=black" alt="Kaggle">
</a>
<a href="mailto:tolgasa2@gmail.com">
    <img src="https://img.shields.io/badge/-Email-red?style=flat-square&logo=gmail&logoColor=white">
</a>
<br/>
<a href="https://github.com/dfavenfre">
    <img src="https://github-stats-alpha.vercel.app/api?username=dfavenfre&cc=22272e&tc=37BCF6&ic=fff&bc=0000">
</a>
</p>

* 💻 AI & ML Engineer building production-grade multimodal AI systems, autonomous agents, and NLP pipelines.
* 📖 Independent research on **morphologically-aware neural tokenization**, **word representations**, and **Retrieval-Augmented Generation** for low-resource / agglutinative languages — under [lonewolf-rd](https://github.com/lonewolf-rd).

---

### 📰 Publications

**Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish** — *arXiv preprint, 2026 (sole author).*
A **lossless, morphology-aware neural tokenizer *and* word embedder** for Turkish. A differentiable Poisson–binomial soft segmentation produces exact, surface-preserving morpheme splits (`decode(encode(w)) = w`), while the same forward pass yields structured word embeddings. Achieves the **lowest BPC among reversible tokenizers** and roughly **2× the morphological alignment** of BPE/WordPiece/Unigram, and leads BERTurk and BGE-M3 on lexical retrieval.
[Repo](https://github.com/lonewolf-rd/TurkishMorpheus) · [Model](https://huggingface.co/lonewolflab/Morpheus-TR-50K) · [Demo](https://huggingface.co/spaces/lonewolflab/morpheus-tr-demo) · *arXiv link coming soon*

**Maximizing RAG Efficiency: A Comparative Analysis of RAG Methods** — *Natural Language Processing, Cambridge University Press (SCI Q1), 2025.*
A grid-search study of **23,625 configurations** across vector stores, embedding models, and LLMs on cross-domain data, quantifying the trade-offs between retrieval quality, similarity-based ranking, token usage, runtime, and hardware utilization. Shows that **contextual compression filters** substantially reduce token consumption and hardware load, at a similarity cost that is often acceptable depending on the RAG method and use case.
[Paper](https://www.cambridge.org/core/journals/natural-language-processing/article/maximizing-rag-efficiency-a-comparative-analysis-of-rag-methods/D7B259BCD35586E04358DF06006E0A85) · [PDF](https://github.com/dfavenfre/dfavenfre/blob/main/maximizing-rag-efficiency-a-comparative-analysis-of-rag-methods.pdf)

---

### 🖥️ Open-Source Projects
<table>
<tr><th>Machine Learning / AI Agents</th></tr>
<tr><td>

|Title | Tech Stack|
|--|--|
| [Multi-Modal RAG](https://github.com/dfavenfre/MultiModal-RAG)| ![LangChain](https://img.shields.io/badge/LangChain-black?style=flat-square&logo=langchain) ![ChromaDB](https://img.shields.io/badge/ChromaDB-green?style=flat-square&logo=chromadb)|
|[RAG Optimization](https://github.com/dfavenfre/RAG-Optimization)| ![LangChain](https://img.shields.io/badge/LangChain-black?style=flat-square&logo=langchain) ![LangSmith](https://img.shields.io/badge/LangSmith-black?style=flat-square&logo=langsmith) ![FAISS](https://img.shields.io/badge/FAISS-black?style=flat-square&logo=faiss)|
|[TalkYou](https://github.com/dfavenfre/TalkYou)| ![LangChain](https://img.shields.io/badge/LangChain-black?style=flat-square&logo=langchain) ![LangGraph](https://img.shields.io/badge/LangGraph-black?style=flat-square&logo=langgraph) ![Docker](https://img.shields.io/badge/Docker-blue?style=flat-square&logo=docker) ![Streamlit](https://img.shields.io/badge/Streamlit-white?style=flat-square&logo=streamlit) ![FastAPI](https://img.shields.io/badge/FastAPI-black?style=flat-square&logo=fastapi)|
| [LLMRoboFund](https://github.com/dfavenfre/LLMRoboFund)| ![LangChain](https://img.shields.io/badge/LangChain-black?style=flat-square&logo=langchain) ![SQL](https://img.shields.io/badge/SQL-green?style=flat-square&logo=sql) ![Streamlit](https://img.shields.io/badge/Streamlit-white?style=flat-square&logo=streamlit) ![ChromaDB](https://img.shields.io/badge/ChromaDB-green?style=flat-square&logo=chromadb)|
|[Electricity Price Forecasting](https://github.com/dfavenfre/electricity-price-forecasting) | ![TF](https://img.shields.io/badge/TF-black?style=flat-square&logo=tensorflow) ![XGBoost](https://img.shields.io/badge/XGBoost-black?style=flat-square)|
|[Olivetti Face Recognition (CNN)](https://github.com/dfavenfre/Olivetti-Faces-PyTorch)| ![PyTorch](https://img.shields.io/badge/PyTorch-black?style=flat-square&logo=pytorch)|
|[Fashion MNIST](https://github.com/dfavenfre/Fashion-MNIST-Tensorflow) | ![W&B](https://img.shields.io/badge/W%26B-black?style=flat-square&logo=wandb) ![TF](https://img.shields.io/badge/TF-black?style=flat-square&logo=tensorflow)|
|[MobileNetV1 Julia Implementation](https://github.com/dfavenfre/MobileNet-Julia)| ![W&B](https://img.shields.io/badge/W%26B-black?style=flat-square&logo=wandb) ![Julia](https://img.shields.io/badge/Julia-black?style=flat-square&logo=julia) ![Flux](https://img.shields.io/badge/Flux-black?style=flat-square&logo=flux)|
| [EfficientNetV2 Transfer Learning (CNN)](https://github.com/dfavenfre/Transfer-Learning-CNN-Fine-Tuning)| ![TF](https://img.shields.io/badge/TF-black?style=flat-square&logo=tensorflow) ![W&B](https://img.shields.io/badge/W%26B-black?style=flat-square&logo=wandb)|
| [Food Vision (CNN)](https://github.com/dfavenfre/Food-Vision-Tensorflow)| ![TF](https://img.shields.io/badge/TF-black?style=flat-square&logo=tensorflow) ![W&B](https://img.shields.io/badge/W%26B-black?style=flat-square&logo=wandb)|
|[Econ Dashboard](https://github.com/dfavenfre/Econ-Dashboard)| ![TF](https://img.shields.io/badge/TF-black?style=flat-square&logo=tensorflow) ![Streamlit](https://img.shields.io/badge/Streamlit-white?style=flat-square&logo=streamlit) ![SQL](https://img.shields.io/badge/SQL-green?style=flat-square&logo=sql)|
|[Bitcoin Price Forecasting](https://github.com/dfavenfre/Bitcoin-Price-Forecasting)| ![PMDARIMA](https://img.shields.io/badge/PMDARIMA-black?style=flat-square) ![SCIPY](https://img.shields.io/badge/SCIPY-black?style=flat-square&logo=scipy)|
|[Bike Sharing Demand Prediction](https://github.com/dfavenfre/Bike-Sharing-Demand-Prediction) | ![XGBoost](https://img.shields.io/badge/XGBoost-black?style=flat-square) ![LGBM](https://img.shields.io/badge/LGBM-black?style=flat-square) ![OPTUNA](https://img.shields.io/badge/OPTUNA-blue?style=flat-square&logo=optuna) ![SCIKITLEARN](https://img.shields.io/badge/SCIKIT-LEARN-blue?style=flat-square&logo=scikit-learn)|
| [Financial Sentiment Classifier](https://github.com/dfavenfre/financial-sentiment-classifier)| ![TF](https://img.shields.io/badge/TF-black?style=flat-square&logo=tensorflow)|
| [Bank Customer Deposit Prediction](https://github.com/dfavenfre/customer_deposit_classifier)| ![XGBoost](https://img.shields.io/badge/XGBoost-black?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-white?style=flat-square&logo=streamlit) ![SCIKITLEARN](https://img.shields.io/badge/SCIKIT-LEARN-blue?style=flat-square&logo=scikit-learn)|
| [Credit Score Prediction](https://github.com/dfavenfre/Credit-Score-Prediction)| ![SCIKITLEARN](https://img.shields.io/badge/SCIKIT-LEARN-blue?style=flat-square&logo=scikit-learn)|

</td></tr>
</table>

<br>

#### 📊 GitHub Stats

![](http://github-profile-summary-cards.vercel.app/api/cards/profile-details?username=dfavenfre&theme=dracula)

![](http://github-profile-summary-cards.vercel.app/api/cards/repos-per-language?username=dfavenfre&theme=dracula)
![](http://github-profile-summary-cards.vercel.app/api/cards/most-commit-language?username=dfavenfre&theme=dracula)

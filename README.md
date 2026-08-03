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

## Contact

Open to research collaboration on tokenization, representation learning and retrieval for morphologically rich languages — [tolgasa2@gmail.com](mailto:tolgasa2@gmail.com)

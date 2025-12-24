# Semantic Video Search Engine 🔍

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![SigLip](https://img.shields.io/badge/Model-Google%20SigLip-green)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-yellow)

A high-performance, AI-driven video search engine that allows you to search for specific moments within videos using natural language queries. Built with state-of-the-art multimodal embeddings and efficient vector indexing.

---

## 🚀 Key Features

- **Natural Language Search**: "Find the moment where the car crashes" or "Show me someone cooking pasta".
- **Advanced Multimodal Embeddings**: Powered by **Google's SigLip (Sigmoid Loss for Language Image Pre-Training)** for superior text-image alignment.
- **Temporal & Semantic Filtering**: Intelligent search pipeline that understands context and filters results by relevance and time.
- **Hybrid Storage Architecture**: Combines **FAISS** (Facebook AI Similarity Search) for blazing fast vector retrieval with **SQLite** for structured metadata management.
- **Variable Frame Extraction**: Supports 'Fast', 'Accurate', and '1fps' extraction modes to balance speed vs. granularity.

---

## ⚡ Performance

> **Benchmark Case**: Tested on Google Colab (T4 GPU).

| Metric | Result |
| :--- | :--- |
| **Input Video Length** | 25 Minutes |
| **Processing Time** | **21 Seconds** ⚡ |
| **Workflow** | Upload ➔ Frame Extraction ➔ Embedding ➔ Indexing ➔ Ready to Search |

*Currently optimizing the pipeline further to remove existing bottlenecks and reduce latency even more.*

---

## 🛠️ Tech Stack

- **Core AI Model**: [Google SigLip](https://huggingface.co/google/siglip-base-patch16-224) (Transformer-based Vision-Text Model)
- **Vector Search Engine**: [FAISS](https://github.com/facebookresearch/faiss)
- **Backend Framework**: Python (Native)
- **Data Management**: SQLite3
- **Image Processing**: Pillow (PIL)

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU recommended for optimal performance (works on CPU with high latency).

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aniket-16-S/Semantic_Video_Search.git
   cd Semantic_Video_Search
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To start the interactive CLI tool:

```bash
python app.py
```

**Workflow:**
1. Select **"2. Add Videos"** to ingest your video files or folders.
2. Choose your extraction method (`fast`, `accurate`, or `1fps`).
3. Once indexed, select **"1. Search"** and type your query!

---

## 🗺️ Roadmap

- [x] Core Search Pipeline
- [x] High-Speed Indexing (SigLip + FAISS)
- [ ] **API Layer**: Expose functionality via REST API for web/mobile integration.
- [ ] **Bottleneck Optimization**: Refactoring the data loader for even faster throughput.
- [ ] **Web UI**: A modern frontend to replace the current CLI.

---

*Project maintained by [Aniket-16-S](https://github.com/Aniket-16-S). currently open for contributions.*

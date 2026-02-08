# 🧠 Advanced Optimization & Regression Analysis on Turkish NLP Data

> A comparative study of various optimization algorithms (SGD, Adam, Newton-Raphson) applied to regression problems derived from Turkish LLM embeddings, featuring dual-scale dataset analysis.

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/AI-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Optimization](https://img.shields.io/badge/Topic-Optimization_Algorithms-green?style=for-the-badge)

## 📖 Project Overview
This project explores the intersection of **Natural Language Processing (NLP)** and **Numerical Optimization**. It involves generating semantic embeddings from Turkish datasets using state-of-the-art Large Language Models (LLMs) and analyzing the performance of different optimization algorithms on regression tasks.

The study aims to observe how different optimizers converge and perform on high-dimensional data reduced via **t-SNE** and **PCA**.

## 🏗️ Project Architecture (Module A & B)
The project is structured into two distinct analytical modules to isolate algorithmic performance from model application:

### 🔹 Module A: Optimization Engine (`Proje_A.py`)
This module focuses on the **manual implementation** and **benchmarking** of fundamental optimization algorithms.
* **Core Task:** Solving regression problems using custom-built optimizers.
* **Algorithms Implemented:** Gradient Descent (GD), Stochastic Gradient Descent (SGD), Adam, RMSProp, Adagrad, and Newton-Raphson.
* **Objective:** To visualize and compare the *convergence speed* and *loss reduction* of each algorithm on the embedding vectors.

### 🔹 Module B: Regression Analysis & Testing (`Proje_B.py`)
This module applies the best-performing optimizers to a broader scope, focusing on **generalization** and **prediction**.
* **Core Task:** Training the regression model on the training set and evaluating performance on the `test_veri_seti.csv`.
* **Focus:** Dimensionality reduction (t-SNE/PCA) and analyzing the model's ability to predict semantic similarity scores on unseen data.

## 📊 Dual-Scale Dataset Strategy
To rigorously test the **scalability** and **computational efficiency** of the optimization algorithms, the project utilizes a dual-dataset approach:

| Dataset | Filename | Purpose |
| :--- | :--- | :--- |
| **Small Scale** | `genel_kultur_veri_seti_kucuk.csv` | Used for rapid prototyping, debugging, and observing initial convergence behaviors without heavy computational load. |
| **Large Scale** | `genel_kultur_veri_seti_buyuk.csv` | **Stress Testing:** Used to evaluate how algorithms (especially Hessian-based ones like Newton-Raphson) handle large-scale matrix operations and high-dimensional data. |

> *Both datasets contain Turkish General Culture QA pairs processed into vector embeddings.*

## ⚙️ Key Methodologies

### 1. Embedding Generation
Semantic vector representations were generated using advanced Turkish LLMs:
* **Cosmos-9B (Turkish-Gemma-9b):** Used for capturing deep semantic relationships.
* **E5 Embeddings:** Utilized for high-quality text representation.

### 2. Optimization Algorithms
The project manually implements the following to solve the regression loss function:
* **Gradient Descent (GD)**
* **Stochastic Gradient Descent (SGD)**
* **Adam Optimizer**
* **Newton-Raphson Method**

## 📂 File Structure
* `Proje_A.py`: Core optimization algorithms and benchmarking.
* `Proje_B.py`: Regression testing and model evaluation.
* `Cosmos9B.py` / `CosmosE5.py`: Embedding generation scripts.
* `data/`: Contains the large and small datasets.
* `DiffÖdev.pdf`: **Full Project Report** (Contains detailed visual analysis and screenshots).

## 📊 Visualizations
> **Note:** Comprehensive performance graphs, loss convergence curves, and t-SNE visualizations are detailed in the **Project Report**. Please refer to the PDF file for visual analysis.

📄 **[Click here to view the Project Description (PDF)](Project.pdf)**
📄 **[Click here to view the Full Report (PDF)](Report.pdf)**

---
**Developed by Ahmet Efe Karahan as part of the BLM2642 Differential Equations for Computer Engineers course curriculum at Yildiz Technical University.**

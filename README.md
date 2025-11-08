# Predicting-Diabetes-Status-from-Lifestyle-and-Health-Indicators-with-Spark
This project explores how Apache Spark MLlib can be used to predict diabetes status based on lifestyle and health indicators from a large dataset. The work compares four supervised machine learning algorithms under both single-node (1 VM) and distributed (2 VM) environments, analyzing performance, accuracy, and scalability.

---

## 🎯 Objectives
- Build predictive models using **Spark MLlib** for large-scale health data.  
- Evaluate four algorithms — Random Forest, Logistic Regression, Gradient Boosting Trees, and Multi-Layer Perceptron.  
- Measure accuracy, AUC, F1-score, and training time.  
- Compare **single-node vs. distributed** Spark setups to evaluate scalability and overhead.  
- Determine performance tradeoffs between algorithm complexity and cluster size.

---

## 🧠 Dataset
**Source:** [Diabetes Health Indicators Dataset – Kaggle](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook)  
**Records:** 253,680  
**Features:** 21 attributes — BMI, blood pressure, cholesterol, physical activity, smoking, and more.  
**Target Variable:** Binary classification (Diabetic vs. Non-diabetic)

This dataset, derived from the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015), provides anonymized survey-based health indicators. It is widely used for public health analytics and early disease prediction.

---

## ⚙️ Infrastructure Setup

| Component | Specification |
|------------|---------------|
| **Framework** | Apache Spark MLlib |
| **Version** | Spark 3.5.0 + Hadoop 3 |
| **Cluster Setup** | 2 Virtual Machines (1 master + 1 worker) |
| **VM Specs** | 1 CPU core, 2 GB RAM |
| **OS** | Fedora Linux |
| **Language** | Python 3.10 (PySpark API) |

---

## 🧩 Data Preparation
- Import dataset from Kaggle into Spark using `spark.read.csv()`  
- Handle missing data and encode categorical variables  
- Assemble features into a single vector using `VectorAssembler`  
- Standardize features with `StandardScaler`  
- Split dataset into **70% training**, **15% validation**, and **15% test** sets  

---

## 🧠 Machine Learning Models

| Algorithm | Category | Key Parameters |
|------------|-----------|----------------|
| **Random Forest** | Ensemble | 100 trees, maxDepth=5, seed=42 |
| **Logistic Regression** | Linear | L2 regularization, maxIter=100 |
| **Gradient Boosting Trees** | Ensemble | 20 iterations, maxDepth=4, stepSize=0.1 |
| **Multi-Layer Perceptron (MLP)** | Neural Network | Layers [21, 32, 16, 2], maxIter=50 |

---

## 📊 Experimental Results

| Algorithm | Nodes | Accuracy | AUC | F1-Score | Train Time (s) |
|------------|--------|----------|------|-----------|----------------|
| Random Forest | 1 VM | 86.80% | 83.57% | 82.55% | 103.25 |
| Random Forest | 2 VMs | 86.80% | 83.57% | 82.55% | 107.26 |
| Logistic Regression | 1 VM | 86.56% | 81.41% | 82.27% | 13.43 |
| Logistic Regression | 2 VMs | 86.56% | 81.41% | 82.27% | 14.16 |
| Gradient Boosting | 1 VM | 86.55% | 81.75% | 83.51% | 66.91 |
| Gradient Boosting | 2 VMs | 86.55% | 81.75% | 83.51% | 75.35 |
| MLP | 1 VM | 86.61% | 81.85% | 81.98% | 47.56 |
| MLP | 2 VMs | 86.61% | 81.85% | 81.98% | 49.84 |

> **Observation:**  
> Performance gains were limited due to small cluster size and communication overhead. Logistic Regression trained fastest, while Random Forest achieved the highest stability.

---

## 📈 Key Insights
- **Accuracy Plateau:** All algorithms achieved ~86%, suggesting diminishing returns without deeper feature engineering.  
- **Computation Overhead:** Distributed setup (2 VMs) slightly slower (4–11%) due to serialization and coordination cost.  
- **Efficiency:** Logistic Regression performed 7–8× faster than Random Forest.  
- **Scalability Threshold:** Spark’s distributed advantages emerge with **larger datasets** (≥ 1 million records).  

---

## 💡 Recommendations
| Area | Suggestion |
|------|-------------|
| **Dataset Size** | Use Spark for ≥ 1M rows; for smaller data, single-node Python (scikit-learn) is faster. |
| **Compute Power** | Allocate ≥ 4 cores per node to reduce communication overhead. |
| **Network Speed** | Use high-throughput connections (>10 Gbps). |
| **Algorithm Choice** | Start with simpler models (Logistic Regression) before complex ensembles. |

---

## 🚀 Future Improvements
- Incorporate feature selection using PCA or Chi-Square filtering.  
- Apply hyperparameter tuning (`CrossValidator` or `TrainValidationSplit`).  
- Extend dataset to multiple years of BRFSS for temporal analysis.  
- Evaluate GPU-based alternatives (e.g., TensorFlow on Dataproc).  
- Benchmark Spark against Dask and Ray for scalability analysis.

---

## 📁 Repository Structure

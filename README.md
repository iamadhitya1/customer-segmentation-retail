# 🛍️ Customer Segmentation for a Retail Company

A Machine Learning project that applies **K-Means Clustering** to segment retail customers based on their Annual Income and Spending Score — built as a minor project during the **AI Internship at SmartED Innovations** (Dec 2025).

---

## Problem Statement

Retail businesses serve thousands of customers with very different behaviours. Without segmentation, marketing is generic and inefficient. This project groups customers into distinct segments so a business can target each group with the right strategy — for example, rewarding high-income/high-spending customers differently from budget-conscious ones.

---

## How It Works

The pipeline follows 6 clear steps:

| Step | Description |
|------|-------------|
| 1 | Load the retail dataset (200 customers, 5 features) |
| 2 | Select features: `Annual Income (k$)` and `Spending Score (1-100)`, then standardize |
| 3 | Use the **Elbow Method** to find the optimal number of clusters |
| 4 | Apply **K-Means** clustering with `k=5` |
| 5 | Visualize clusters using a Seaborn scatter plot |
| 6 | Analyze each cluster's mean income and spending score |

---

## Dataset

**File:** `retail_data.csv`

| Column | Description |
|--------|-------------|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male / Female |
| `Age` | Customer age (18–70) |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Score assigned by the store based on spending behaviour |

200 rows · 5 columns

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white)

- **Algorithm:** K-Means Clustering (`sklearn.cluster.KMeans`)
- **Preprocessing:** StandardScaler (`sklearn.preprocessing`)
- **Visualization:** Matplotlib (Elbow curve) + Seaborn (cluster scatter plot)
- **Concepts:** Unsupervised Learning, Clustering, Feature Scaling, Elbow Method

---

## Project Structure

```
customer-segmentation-retail/
├── customer_segmentation.py     # Full ML pipeline
├── retail_data.csv              # Customer dataset (200 rows)
├── Report for Minor Project.pdf # Full project report (SmartED Innovations)
└── README.md
```

---

## How to Run

### Prerequisites
- Python 3.x

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the project
```bash
python customer_segmentation.py
```

### What you'll see
1. First 5 rows of the dataset printed to terminal
2. **Elbow Method graph** — helps identify optimal `k`
3. **Customer Segments scatter plot** — 5 colour-coded clusters
4. **Cluster analysis table** — mean income and spending score per cluster

---

## Key Concepts

**K-Means Clustering** partitions data into `k` groups where each point belongs to the cluster with the nearest centroid. It minimises Within-Cluster Sum of Squares (WCSS).

**The Elbow Method** plots WCSS against number of clusters. The "elbow" point — where the curve bends — indicates the optimal `k`. Here that's `k=5`.

**StandardScaler** normalises features to have mean=0 and std=1, ensuring Annual Income (large numbers) doesn't dominate Spending Score (1–100 range) during clustering.

---

## Results

5 customer segments are identified, roughly corresponding to:

| Segment | Income | Spending | Profile |
|---------|--------|----------|---------|
| 0 | High | High | Premium customers — top priority |
| 1 | Low | High | Enthusiastic but budget-constrained |
| 2 | Medium | Medium | Average customers |
| 3 | High | Low | Wealthy but not engaged |
| 4 | Low | Low | Price-sensitive, low engagement |

*(Exact cluster labels vary per run due to K-Means random initialisation)*

---

## What I Learned

- How unsupervised learning differs from supervised — no labels, patterns emerge from data
- Why feature scaling is critical before distance-based algorithms like K-Means
- How the Elbow Method guides hyperparameter selection (`k`)
- Translating cluster output into real business insight

---

## Context

Built during the **AI Internship at SmartED Innovations** (December 2025) as the minor project submission. Full report included as `Report for Minor Project.pdf`.

---

## Author

**M. Adhitya** — B.Tech Computer Engineering, IITRAM Ahmedabad
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/loveadhitya/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/iamadhitya1)

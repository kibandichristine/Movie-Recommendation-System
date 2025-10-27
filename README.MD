# 🎭 MovieLens Puppet Theater Recommender  
**Improving user experience through the development of an advanced movie recommendation system.**



---

## 🌐 Live Demo  
🚀 **Experience the Movie Recommender App here:**  
👉 [**MovieLens Puppet Theater — Streamlit App**](https://kibandichristine-movie-recommendation-system-app-pbwohz.streamlit.app/?embed_options=show_toolbar,show_colored_line,show_padding,show_footer,dark_theme)

---

## 📘 Business Understanding  

### **Introduction**  
In today's vast world of movies, choosing what to watch can be overwhelming. This project addresses that challenge by building a **Movie Recommendation System** that enhances user experience through **personalized movie suggestions** using the **MovieLens dataset**.

### **Problem Statement**  
The goal is to develop an **intelligent recommendation system** that mitigates choice overload. By leveraging data-driven insights, the system simplifies the movie selection process and increases user satisfaction.

### **Dataset Overview**  
The **MovieLens dataset (ml-latest-small)** contains:  
- **100,836 ratings**  
- **3,683 tag applications**  
- **9,742 movies**  
- **610 users** (each with ≥ 20 ratings)  

All data was collected between **March 29, 1996** and **September 24, 2018**, with consistent user and movie IDs across all CSV files:  
- `ratings.csv`  
- `movies.csv`  
- `tags.csv`  
- `links.csv`  

---

## 🧩 Dataset Structure  

| File | Format | Description |
|------|---------|-------------|
| **ratings.csv** | `userId, movieId, rating, timestamp` | User ratings on a 5-star scale |
| **tags.csv** | `userId, movieId, tag, timestamp` | User-generated descriptive tags |
| **movies.csv** | `movieId, title, genres` | Movie titles and genres (pipe-separated) |
| **links.csv** | `movieId, imdbId, tmdbId` | Cross-links to IMDb and TMDb |

---

## 🎯 Metrics of Success  
- **Click-through rate (CTR)**  
- **User ratings**  
- **Retention rates**  
- **Quality of feedback**

---

## 🏆 Project Objectives  
- Enhance **user experience**  
- Increase **user engagement**  
- Develop **recommendation algorithms**  
- Improve **interaction and feedback** utilization  

---

## 🧠 Data Understanding  
The system leverages **100,000+ user ratings** to identify relationships between movies using correlation analysis and heatmap visualization.

---

## 🧹 Data Preprocessing  
- Converted ratings to numeric format  
- Handled missing values and outliers  
- Created pivot tables (user–movie matrices) for similarity calculations  

---

## 🏗️ Modeling Approach  

### **1. User-Based Collaborative Filtering (UserCF)**  
Recommends movies based on users with similar preferences using **cosine similarity**.

### **2. Item-Based Collaborative Filtering (ItemCF)**  
Suggests movies that are similar to ones a user already likes.

### **3. Matrix Factorization (SVD)**  
Uses **Singular Value Decomposition** to predict missing ratings and capture hidden user–movie features.

### **4. Hybrid Model**  
Combines **UserCF** and **ItemCF** through a weighted approach for improved accuracy.

### **5. Popularity-Based Recommendations**  
Provides suggestions for new users (cold start problem) based on overall movie popularity.

### **6. Content-Based Recommendations**  
Generates recommendations based on **movie genres**.

---

## 🚀 Getting Started  

### **Prerequisites**  
- Python **3.8+**  
- Required libraries:
  ```bash
  pip install streamlit pandas numpy scikit-learn scipy

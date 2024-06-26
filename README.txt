# Improving user experience through the development of an advanced movie recommendation system.

## Business Understanding

### Introduction

In today's vast world of movies, the overwhelming task of movie selection is addressed by the Movie Recommendation System. Aiming to enhance user experiences by suggesting personalized movie choices, this project utilizes the MovieLens dataset.

### Problem Statement

The challenge posed by this project is the development of an intelligent recommendation system to mitigate the issue of choice overload for users. The goal is to simplify the movie selection process by providing data-driven movie recommendations.


## Overview
The MovieLens dataset (ml-latest-small) provides insights into 5-star rating and free-text tagging activities from MovieLens, a popular movie recommendation service. The dataset encompasses 100,836 ratings and 3,683 tag applications across 9,742 movies. These records were generated by 610 users from March 29, 1996, to September 24, 2018. The dataset version being referred to was created on September 26, 2018.

Users were randomly selected for inclusion in the dataset, with a prerequisite that each user had rated at least 20 movies. No demographic information about the users is included. A unique ID represents each user, and no other user-specific data is provided.

The dataset is composed of four CSV files: `links.csv`, `movies.csv`, `ratings.csv`, and `tags.csv`. Detailed information about the contents and the utilization of these files is provided below.

Dataset Structure and Information

### 1: User IDs
- User IDs are anonymized and consistent across `ratings.csv` and `tags.csv`.

### 2: Movie IDs
- Only movies with at least one rating or tag are included in the dataset.
- Movie IDs are consistent with those used on the MovieLens website.
- Consistency across `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv`.

### 3: Ratings Data (ratings.csv)
- Format: userId, movieId, rating, timestamp
- Ratings are on a 5-star scale with half-star increments (0.5 stars - 5.0 stars).
- Timestamps are in seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

### 4:Tags Data (tags.csv)
- Format: userId, movieId, tag, timestamp
- Tags are user-generated metadata about movies.
- Timestamps are in seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

### 5: Movies Data (movies.csv)
- Format: movieId, title, genres
- Movie titles include the release year in parentheses.
Genres are pipe-separated.

6: Links Data (links.csv)
- Format: movieId, imdbId, tmdbId
- Links to other movie data sources.


## Metric of Success

The primary metrics of success are as follows:
- Click-through rates
- User ratings
- Retention rates
- Feedback quality

## Objectives

The objectives of this project include:
- Enhancement of User Experience
- Increase in User Engagement
- Development of Recommendation Algorithms
- Enhancement of User Interaction
- Utilization of Feedback

## Data Understanding

The foundation of this recommendation system is the MovieLens dataset, consisting of 100,000 user ratings. Understanding the dataset's structure and characteristics is considered vital for the construction of an effective recommendation model.

## Correlation Matrix

The correlation matrix is calculated to observe the correlation between movie ratings. Visualization of the correlation matrix is done via a heatmap.

## Data Preprocessing

Data preprocessing is conducted, encompassing the conversion of ratings into a numeric format and the handling of outliers.

## Creation of Movie Matrix

A pivot table is created to form a movie matrix, a crucial element for identifying similarities between movies and users based on ratings.

## User Ratings Input

Users are provided with the capability to input their movie ratings, which are then used to generate recommendations.

## Modeling

Collaborative filtering techniques are employed for movie recommendations.

## User-Based Collaborative Filtering

Movies are recommended to users based on the preferences and behaviors of users with similarities to them. User-user similarity is calculated for making personalized recommendations.

## Item-Based Collaborative Filtering

Movies are recommended to users based on the similarity between movies. Item-item similarity is computed for personalized recommendations.

## Matrix Factorization (SVD)

Singular Value Decomposition (SVD) is used for matrix factorization to create a recommendation model and generate predicted ratings.

## Hyperparameter Tuning

Hyperparameters are fine-tuned to optimize the performance of the recommendation model.

## Hybrid Recommendation System

A weighted hybrid approach is employed, combining user-based and item-based collaborative filtering, for more accurate and diverse recommendations.

## Popularity-Based Recommendations

Recommendations are provided for new users based on the popularity of movies.

## Content-Based Recommendations

Recommendations are made based on movie genres.

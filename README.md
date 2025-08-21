# Book Recommendation System: Predicting Your Next Best Read

**Group W04G9** – Emily Joseph, Erich Wiguna, Raphael Renaldo  

---

## Overview
This project implements a recommendation system to predict how users would rate new books based on their past ratings. Using collaborative filtering, TF-IDF vectorization of book titles, and K-Means clustering, the system provides personalized book recommendations.

---

## Methodology
- **Data Preprocessing:** Removed outliers in Ratings, Age, and Year; imputed missing ages using KNN with geographic coordinates.  
- **Feature Engineering:** Only User ID and book title clusters were used due to high correlation with ratings.  
- **TF-IDF & Clustering:** Book titles were processed and clustered using K-Means to create similarity groups.  
- **Rating Prediction:** Predicted using user-cluster mean ratings; fallback to user overall mean if no cluster match existed.

---

## Results
- **Accuracy:** 62.4% (baseline 30%)  
- Model predicts high ratings well, but medium and low ratings less accurately due to skewed title clusters and sparse user history.

---

## Limitations & Future Improvements
- Skewed title cluster distribution  
- Few books read per user limits predictions  
- Missing book genres and user preferences  
- Future improvements: incorporate genres, user-based collaborative filtering, and expand dataset coverage

---

## Conclusion
The system effectively predicts user ratings for new books, providing personalized recommendations to enhance engagement and support authors and publishers.

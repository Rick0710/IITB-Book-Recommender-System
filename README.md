# IITB Book Recommender System

## Project Overview

This project, developed as part of coursework at IIT Bombay, focuses on building a book recommendation system using machine learning and collaborative filtering techniques. The aim is to provide personalized book recommendations to users based on their reading preferences, leveraging user ratings and book metadata.

## Work Done

### 1. **Data Collection and Preprocessing**
- **Dataset**: Utilized the Goodreads dataset, containing user ratings, book titles, authors, genres, and metadata.
- **Preprocessing Steps**:
  - Cleaned book metadata by standardizing titles, author names, and genres.
  - Handled missing ratings by imputing with user or book average ratings.
  - Filtered out users with fewer than 5 ratings and books with fewer than 10 ratings to ensure data quality.
  - Created a user-item matrix for collaborative filtering, with rows as users and columns as books.

### 2. **Feature Engineering**
- **Content-Based Features**:
  - Extracted TF-IDF vectors from book descriptions and genres for content-based filtering.
  - Used pre-trained word embeddings (e.g., Word2Vec) to capture semantic similarities between book descriptions.
- **Collaborative Filtering Features**:
  - Computed user-user and item-item similarities using cosine similarity and Pearson correlation.
  - Generated features like average rating per user and book popularity scores.

### 3. **Model Development**
- **Algorithms Explored**:
  - **K-Nearest Neighbors (KNN)**: Used for user-based and item-based collaborative filtering.
  - **Matrix Factorization**: Implemented Singular Value Decomposition (SVD) to reduce dimensionality of the user-item matrix.
  - **Neural Collaborative Filtering (NCF)**: Built a neural network combining matrix factorization and deep learning for hybrid recommendations.
- **Model Architecture**:
  - NCF: Combined Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) to learn user-item interactions.
  - Hyperparameters tuned: latent factors, learning rate, dropout rate, and batch size.
- **Training**:
  - Split dataset into 80% training, 10% validation, and 10% test sets.
  - Used Mean Squared Error (MSE) loss and Adam optimizer.
  - Applied regularization to prevent overfitting in NCF.

### 4. **Evaluation**
- **Metrics**:
  - Root Mean Squared Error (RMSE): Measured prediction accuracy for ratings.
  - Precision@K and Recall@K: Evaluated top-K recommendation quality.
  - Coverage: Assessed the fraction of books the system could recommend.
- **Results**:
  - KNN (User-Based): RMSE ~0.95, Precision@10 ~0.65.
  - SVD: RMSE ~0.88, Precision@10 ~0.70.
  - NCF: RMSE ~0.85, Precision@10 ~0.73 (best performing model).

### 5. **Web Application**
- Developed a Flask-based web app to demonstrate the recommendation system.
- **Functionality**:
  - Users input their user ID or select books they’ve read, and the app returns top-K book recommendations.
  - Displays book details (title, author, genre) and predicted ratings.
- **Implementation**:
  - Integrated the trained NCF model for real-time predictions.
  - Used Bootstrap for front-end styling and SQLite for storing user interactions.

### 6. **Challenges and Solutions**
- **Challenge**: Sparse user-item matrix due to limited ratings.
  - **Solution**: Applied matrix factorization and used content-based features to handle cold-start problems.
- **Challenge**: Scalability issues with large datasets.
  - **Solution**: Optimized matrix operations using sparse matrices and batch processing.
- **Challenge**: Balancing content-based and collaborative filtering.
  - **Solution**: Implemented a hybrid approach in NCF to leverage both user ratings and book metadata.

### 7. **Future Work**
- Incorporate advanced models like BERT for better text feature extraction from book descriptions.
- Add real-time user feedback to improve recommendations dynamically.
- Deploy the web app on a cloud platform like Heroku or AWS for scalability.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Rick0710/IITB-Book-Recommender-System.git
   cd IITB-Book-Recommender-System
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Embeddings**:
   - Download Word2Vec embeddings from [Google Code Archive](https://code.google.com/archive/p/word2vec/) and place them in the `data/` folder.

4. **Run the Web App**:
   ```bash
   flask run
   ```

## Dependencies
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `flask`, `gensim`, `sqlite3`

## References
- Goodreads dataset: [Kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks)

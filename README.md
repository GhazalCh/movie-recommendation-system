# Movie Recommendation System Using LDA and User Similarity

## 📌 Overview

This project implements a movie recommendation system using **Latent Dirichlet Allocation (LDA)** for topic modeling and **cosine similarity** to identify users with similar movie preferences.

The system analyzes users' movie ratings and movie metadata, including titles and genres, to discover latent preference patterns. It then generates personalized movie recommendations based on similar users' highly rated movies.

The project also visualizes relationships between users, topics, and movies as a directed graph and evaluates recommendation performance using Hit Rate@10.

## 🎯 Objectives

* Explore user preferences through topic modeling.
* Represent users using distributions over latent topics.
* Identify similar users using cosine similarity.
* Recommend movies based on the preferences of similar users.
* Visualize user–topic–movie relationships as a directed graph.
* Evaluate recommendation performance using Hit Rate@10.

## ⚙️ Methodology

The project consists of the following stages:

### 1. Data Preparation

The system processes movie rating data and movie metadata.

To reduce computational complexity during experimentation, the preprocessing stage creates a smaller dataset by:

* Selecting the 200 most active users.
* Selecting the 300 most frequently rated movies among the selected users.
* Sampling up to 20 ratings per user.
* Saving the resulting dataset as `mini_ratings.csv`.

A fixed random seed is used when sampling ratings to improve reproducibility.

### 2. Movie Content Representation

Movie titles and genres are combined to create textual representations of movies.

The text is normalized by converting it to lowercase and removing non-word characters.

Movie IDs are also included as tokens. This allows the topic model to associate latent topics with specific movies as well as their textual characteristics.

### 3. User Preference Modeling with LDA

The system constructs a document for each user using the content of movies they have rated. Movie content is repeated according to the corresponding rating value.

A **Latent Dirichlet Allocation (LDA)** model with 15 topics is trained on these user documents using `CountVectorizer`.

The resulting topic distributions represent users' latent preference profiles.

The model also estimates topic–word distributions, which are used to associate topics with movies.

### 4. User Similarity

Cosine similarity is calculated between the normalized topic distributions of users.

Users with similar topic distributions are treated as potential neighbors for recommendation purposes.

### 5. Movie Recommendation

The recommendation process follows these steps:

1. Find the most similar users for a target user.
2. Identify movies rated highly by those neighbors.
3. Exclude movies the target user has already rated.
4. Aggregate neighbor similarity scores for candidate movies.
5. Rank the candidates and return the top recommendations.

The default recommendation threshold considers ratings of 4 or higher as positive preferences.

### 6. Graph Visualization

The project visualizes relationships between users, latent topics, and movies using a directed graph implemented with NetworkX.

The graph contains three types of nodes:

* **Users:** Represent individual users.
* **Topics:** Represent latent preference patterns discovered by LDA.
* **Movies:** Represent individual movies.

Edges connect users to their most prominent topics and topics to movies with high topic-related scores.

Edge weights indicate the corresponding user–topic or topic–movie scores.

### 7. Evaluation

The system evaluates recommendation performance using **Hit Rate@10**.

The ratings are split into training and test subsets, with 20% of the records assigned to the test set.

For each user, the evaluation checks whether at least one held-out movie appears among the top 10 recommendations.

The final metric is the proportion of evaluated users for whom at least one held-out movie is successfully recommended.

## 🛠️ Technologies Used

* **Python** — Core programming language
* **Pandas** — Data manipulation and preprocessing
* **NumPy** — Numerical operations
* **SciPy** — Sparse matrix representation
* **Scikit-learn** — LDA, text vectorization, cosine similarity, and dataset splitting
* **NetworkX** — Graph construction and visualization
* **Matplotlib** — Visualization
* **tqdm** — Progress-bar utilities

## 📂 Project Structure

A recommended repository structure is:

```text
movie-recommendation-lda/
│
├── Project1.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   ├── u.data
│   ├── u.item
│   └── mini_ratings.csv
│
└── .gitignore
```

**Note:** The `data/` directory above is a suggested organization, not the current directory structure required by the notebook. The notebook currently uses absolute Windows paths and must be updated to match the chosen repository structure.

The raw dataset files and generated files should only be included if their redistribution is permitted and they are necessary for reproducing the project.

## 📊 Results

When executed, the notebook:

* Creates a reduced movie-rating dataset.
* Learns latent user preference topics.
* Visualizes an expanded user–topic–movie graph.
* Prints sample neighbor-based movie recommendations with genres.
* Calculates Hit Rate@10 on the held-out ratings.

Actual recommendation examples and evaluation scores should be added here after running the notebook.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-lda.git
cd movie-recommendation-lda
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Dependencies

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the required dataset files in the project's data directory:

* `u.data` — User–movie ratings.
* `u.item` — Movie metadata, including titles and genres.

Update the file paths in the notebook to point to these files.

### 4. Run the Notebook

Open `Project1.ipynb` in Jupyter Notebook or Visual Studio Code and execute its cells in order.

The first cell creates the reduced rating dataset. The second cell runs the recommendation pipeline, generates the graph visualization, prints sample recommendations, and evaluates the model.

## 🔮 Future Improvements

Potential extensions include:

* Replacing absolute file paths with configurable relative paths.
* Improving evaluation by ensuring recommendations are generated without access to held-out ratings.
* Comparing LDA-based user representations with alternative recommendation methods.
* Incorporating additional movie metadata.
* Experimenting with different numbers of latent topics.
* Evaluating the system using additional ranking metrics such as Precision@K, Recall@K, and NDCG@K.
* Improving the recommendation strategy by incorporating rating values directly into candidate ranking.

## 👩‍💻 Author

**Ghazal Cheraghi**

This project was developed as an academic project exploring topic modeling, user similarity, graph visualization, and personalized movie recommendation.

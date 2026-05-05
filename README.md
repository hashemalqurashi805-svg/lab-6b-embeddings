# lab-6b-embeddings
Semantic Space Visualization: Exploring Word & Document Embeddings
📌 Project Overview
This project focuses on visualizing high-dimensional text data in a 2D plane using t-SNE (t-Distributed Stochastic Neighbor Embedding). It explores the semantic relationships between individual words using GloVe-style vectors and full news articles using the DistilBERT transformer model.

The goal is to demonstrate how machine learning models "understand" context by clustering similar concepts together in a vector space.

🛠️ Technical Stack
Language: Python 3.x.

Environment: Windows Subsystem for Linux (WSL2 / Ubuntu).

Libraries:

Transformers (Hugging Face) for DistilBERT embeddings.

Scikit-learn for t-SNE dimensionality reduction.

Pandas & NumPy for data manipulation.

Matplotlib for data visualization.

🚀 Key Features
Word Clustering: Visualizing 200 words across 5 distinct categories (Technology, Sports, Countries, Emotions, Business).

Document Analysis: Processing 20 news articles from the BBC News Dataset.

Automated Export: Scripts automatically generate and save plots as .png files for easy analysis.

Optimized Performance: Leveraging 16GB RAM for efficient transformer inference and t-SNE computations.

📊 Visual Results
The project generates two main visualizations:

glove_clusters.png: Shows how words like "software" and "algorithm" group together away from "happy" or "sad".

bbc_document_clusters.png: Illustrates how BBC articles are clustered by their respective categories (e.g., tech vs. sport).

⚙️ Setup & Execution
Activate Virtual Environment:

Bash
source .venv/Scripts/activate
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Explorer:

Bash
python stretch_embedding_explorer.py
👨‍💻 Author
Hashem Al-Qurashi
Lead Developer | AI.SPIRE Trainee
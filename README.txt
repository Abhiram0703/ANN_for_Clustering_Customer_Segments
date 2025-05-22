================================================
     ANN for Clustering Customer Segments
================================================

Course: DA221, Introduction to AI  


The Upside Down, Members:
-------------------------
- M. Abhiram (230150015)
- K. Ashmita (230150014)
- B. Cherish (230150007)

Description:
------------
This project is about clustering customers, based on their purchasing behaviour in a shopping mall.
Unlike K means algorithm, we used an artificial neural network (ANN) to cluster the dataset. 
We also implemented a loss function to make it a supervised learning method, rather than competitive learning.

Implementation:
--------------
- Load and preprocess the dataset
- Exploratory Data Analysis (EDA) of the dataset using matplotlib and seaborn
- Normalize numerical features using StandardScaler from Scikit-learn
- Designing an Autoencoder using PyTorch for Dimensionality Reduction
- Extracting the Latent Spaces from the Bottleneck Layer
- Designing an ANN for Clustering using PyTorch
- Defining MMJ-K-Means Loss for Soft Clustering
- Training the ANN and visualizing the clusters using matplotlib
- Defining Silhouette Coefficient which acts as an internal evaluation index for clustering
- Interpreting the clusters formed from the ANN with the highest Silhoutte Coefficient in terms of marketability

Libraries Used:
------------------
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- PyTorch

Dataset:
--------
- `Mall_Customers.csv` (must be in the same directory as the notebook)
- It contains 5 features : `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`
- `CustomerID` feature is just redundant
- `Gender` is a categorical feature. We use one hot encoding to convert it into a numerical feature
- It contains 200 sample data points

Requirements:
-------------
Install the following Python packages (if not already installed):

pip install pandas numpy matplotlib seaborn scikit-learn torch

How to Run:
-----------
1. Open the Jupyter Notebook file: `ANN_Clustering_Code.ipynb`
2. Make sure `Mall_Customers.csv` is present in the same directory.
3. Run each cell in order from top to bottom.
4. The notebook will:
   - Load and preprocess the data
   - Train the Autoencoder for dimensionality reduction and extracts the Latent Space
   - Train the ANN for clustering
   - Output clustering results with visualizations

Output:
-------
- Exploratory Data Analysis
- Latent Space of the Dataset
- Cluster visualizations by different ANNs
- Insights into targeted Marketing by analyzing each cluster separately.

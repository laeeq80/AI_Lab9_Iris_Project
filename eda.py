# eda.py
# EXPLORATORY DATA ANALYSIS

#Import necessary libraries: We import matplotlib.pyplot and seaborn for visualization.
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    # Pairplot: sns.pairplot creates pairwise scatter plots to visualize relationships
    # between features, colored by species.
    sns.pairplot(data, hue='species', markers=["o", "s", "D"])
    
    # Show plot: plt.show() displays the plot.
    plt.show()
    
    # Feature distribution: data.hist creates histograms for each feature to inspect their distributions.
    data.hist(bins=20, figsize=(12, 10))
    plt.show()

# eda.py
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    sns.pairplot(data, hue='species', markers=["o", "s", "D"])
    plt.show()
    data.hist(bins=20, figsize=(12, 10))
    plt.show()

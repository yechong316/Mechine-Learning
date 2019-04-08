from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
iris = datasets.load_iris()

irisData = pd.DataFrame(iris.data,columns=iris.feature_names)
print(irisData.head(5))

irisData.corr()

plt.figure(figsize=(10,10))
sns.heatmap(irisData.corr(),annot=True)
plt.show()

# plt.figure(figsize=(10,10))
# sns.heatmap(irisData.corr(),annot=True)
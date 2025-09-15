from sklearn import datasets
iris = datasets.load_iris()

import pandas as pd
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)

PlantGrowth.describe()
# 1. Using the iris dataset...

# a. Make a histogram of the variable Sepal.Width.
import matplotlib.pyplot as plt
plt.hist(iris.data[:, 1], bins=10, edgecolor='black')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()
# b. Based on the histogram from #1a, which would you expect to be higher, the mean or the median? Why?
# The mean is is higher than the median because the histogram shows a right-skewed distribution.

# c. Confirm your answer to #1b by actually finding these values.
import numpy as np
print('Mean: ', np.mean(iris.data[:, 1])) # 3.0573333333333337
print('Median: ', np.median(iris.data[:, 1])) # 3.0

# d. Only 27% of the flowers have a Sepal.Width higher than ________ cm.
print('27th percentile: ', np.percentile(iris.data[:, 1], 73)) # 3.3

# e. Make scatterplots of each pair of the numerical variables in iris (There should be 6 pairs/plots).
import seaborn as sns
import pandas as pd
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
sns.pairplot(iris_df)
plt.show()

# f. Based on #1e, which two variables appear to have the strongest relationship? And which two appear to have the weakest relationship?
# The two variables with the strongest relationship are Petal Length and Petal Width. 
# The two variables with the weakest relationship are Sepal Length and Sepal Width.

#2. Using the PlantGrowth dataset...

# a. Make a histogram of the variable weight with breakpoints (bin edges) at every 0.3 units, starting at 3.3.
plt.hist(PlantGrowth['weight'], bins=np.arange(3.3, 6.4, 0.3), edgecolor='black')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

# b. Make boxplots of weight separated by group in a single graph.
sns.boxplot(x='group', y='weight', data=PlantGrowth)
plt.show()

# c. Based on the boxplots in #2b, approximately what percentage of the "trt1" weights are below the minimum "trt2" weight?
trt1_weights = PlantGrowth[PlantGrowth['group'] == 'trt1']['weight']
trt2_min_weight = PlantGrowth[PlantGrowth['group'] == 'trt2']['weight'].min()
percentage_below = (trt1_weights < trt2_min_weight).mean() * 100
print(f'Approximately {percentage_below:.2f}% of the "trt1" weights are below the minimum "trt2" weight.')

# d. Find the exact percentage of the "trt1" weights that are below the minimum "trt2" weight.
exact_percentage_below = (trt1_weights < trt2_min_weight).sum() / len(trt1_weights) * 100
print(f'Exact percentage of "trt1" weights below minimum "trt2" weight: {exact_percentage_below:.2f}%')

# e. Only including plants with a weight above 5.5, make a barplot of the variable group. 
# Make the barplot colorful using some color palette.
filtered_data = PlantGrowth[PlantGrowth['weight'] > 5.5]
sns.countplot(x='group', data=filtered_data, palette='Set1')
plt.show()
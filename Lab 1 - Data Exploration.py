import pandas as pd

df = pd.read_csv(r'C:\Users\f8iqbal\Desktop\iris.data.csv', names=['sepal length', 'sepal width', 'petal length',
'petal width', 'class'])



mean_data = df[["sepal length", "sepal width", "petal length", "petal width"]].mean()
print(mean_data)

max_data = df[["sepal length", "sepal width", "petal length", "petal width"]].max()
print(max_data)

min_data = df[["sepal length", "sepal width", "petal length", "petal width"]].min()
print(min_data)

std_data = df[["sepal length", "sepal width", "petal length", "petal width"]].std()
print(std_data)

distinct_class = df.value_counts(df['class'])
print(distinct_class)


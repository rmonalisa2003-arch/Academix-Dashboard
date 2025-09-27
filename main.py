import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Show all columns and full values
pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.max_colwidth", None)  # show full text in each cell


# Load the dataset
df = pd.read_csv('Student_scores.csv')

# Show first 5 rows
print("\n--- Head ---")
print(df.head())

# Summary statistics
print("\n--- Describe ---")
print(df.describe(include="all"))

# Info
print("\n--- Info ---")
print(df.info())

# Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Drop unwanted column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

print("\n--- After Dropping Unnamed: 0 ---")
print(df.head())

# change WklyStudyHours column 
df["WklyStudyHours"] = df["WklyStudyHours"].str.replace("05-Oct", "5-10")
df.head()

# gender distribution
plt.figure(figsize=(4, 4))
ax = sns.countplot(data=df, x="Gender")
ax.bar_label(ax.containers[0])
plt.show()

# impact of parents education on student scores
gb = df.groupby("ParentEduc").agg({"MathScore": "mean", "ReadingScore": "mean", "WritingScore": "mean"})
print (gb)
plt.figure(figsize=(4, 4))
sns.heatmap(gb, annot=True, cmap="YlGnBu")
plt.show()

# impact of parents marital status on student scores
gb1 = df.groupby("ParentMaritalStatus").agg({"MathScore": "mean", "ReadingScore": "mean", "WritingScore": "mean"})
print (gb1)
plt.figure(figsize=(4, 4))
sns.heatmap(gb1, annot=True, cmap="YlGnBu")
plt.show()

# impact of wklyStudyHours on student scores
gb2 = df.groupby("WklyStudyHours").agg({"MathScore": "mean", "ReadingScore": "mean", "WritingScore": "mean"})
print (gb2)
plt.figure(figsize=(3, 3))
sns.heatmap(gb2, annot=True, cmap="YlGnBu")
plt.show()

# to detect outliers(using box plot)
sns.boxplot(data=df, x="MathScore", y="WklyStudyHours")
plt.show()

sns.boxplot(data=df, x="ReadingScore")
plt.show()

print(df["EthnicGroup"].unique())
# distribution of ethinic groups
groupA = df.loc[df["EthnicGroup"] == "group A"].count()
groupB = df.loc[df["EthnicGroup"] == "group B"].count()
groupC = df.loc[df["EthnicGroup"] == "group C"].count()
groupD = df.loc[df["EthnicGroup"] == "group D"].count()
groupE = df.loc[df["EthnicGroup"] == "group E"].count()

mylist = [groupA["EthnicGroup"], groupB["EthnicGroup"], groupC["EthnicGroup"], groupD["EthnicGroup"], groupE["EthnicGroup"]]
plt.pie(mylist, labels=["group A", "group B", "group C", "group D", "group E"], autopct='%1.1f%%')#autopct used for percentage
plt.show()

# check for actual score value group wise using bar plot
plt.figure(figsize=(4, 4))
ax = sns.countplot(data=df, x="EthnicGroup")
ax.bar_label(ax.containers[0])
plt.show()
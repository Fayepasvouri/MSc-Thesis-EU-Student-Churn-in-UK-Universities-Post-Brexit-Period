"""
Faye

"""

from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')


df = pd.read_csv("C:/Users/Faye/Desktop/Master/data_geo_dissertation.csv")
print(df)

print(df.Q17)
df1 = df.replace(np.nan, '', regex=True)
print(df1)

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in df1.Q17:
    pol_score = sia.polarity_scores(line)
    pol_score['Q17'] = line
    results.append(pol_score)

pprint(results)


df = pd.DataFrame.from_records(results)
df.head()

df['Q17'] = 0
df.loc[df['compound'] > 0.2, 'Q17'] = 1
df.loc[df['compound'] < -0.2, 'Q17'] = -1
df.head()

print("Positive Comments:\n")
pprint(list(df[df['Q17'] == 1].Q17)[:5], width=200)

print("\nNegative Comments:\n")
pprint(list(df[df['Q17'] == -1].Q17)[:5], width=200)

print(df.Q17.value_counts())

print(df.Q17.value_counts(normalize=True) * 100)

fig, ax = plt.subplots(figsize=(8, 8))

counts = df.Q17.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()



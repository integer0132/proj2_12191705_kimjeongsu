import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

file = pd.read_csv('/content/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

user_id = np.arange(1,6041)
movie_id = np.arange(1,3953)

df = pd.DataFrame(index=user_id, columns=movie_id).fillna(0)
for tmp in file.index:
  df.loc[file.loc[tmp, 'UserID'], file.loc[tmp, 'MovieID']] = file.loc[tmp, 'Rating']

df.index.name = 'Users'
df.columns.name = 'Items'

km = KMeans(n_clusters=3, random_state=0, n_init='auto')
km.fit(df)

y_km = km.predict(df)

rank = np.arange(1, 11)
#AU
for AU_tmp in [0, 1, 2]:
  AU = pd.DataFrame(df[y_km == AU_tmp].sum().sort_values(ascending=False)[:10].index, index=rank, columns=['Items'])
  print(f"AU top 10 recommendation for Group {AU_tmp}")
  print(AU)
  print()

#Avg
for Avg_tmp in [0, 1, 2]:
  Avg = pd.DataFrame(((df[y_km == Avg_tmp][df > 0]).mean(0)).sort_values(ascending=False)[:10].index, index=rank, columns=['Items'])
  print(f"Avg top 10 recommendation for Group {Avg_tmp}")
  print(Avg)
  print()

#SC
for SC_tmp in [0, 1, 2]:
  SC = pd.DataFrame(df[y_km == SC_tmp][df > 0].count().sort_values(ascending=False)[:10].index, index=rank, columns=['Items'])
  print(f"SC top 10 recommendation for Group {SC_tmp}")
  print(SC)
  print()

#AV
for AV_tmp in [0, 1, 2]:
  AV = pd.DataFrame(df[y_km == AV_tmp][df > 3].count().sort_values(ascending=False)[:10].index, index=rank, columns=['Items'])
  print(f"AV top 10 recommendation for Group {AV_tmp}")
  print(AV)
  print()

#BC
for BC_tmp in [0, 1, 2]:
  BC = pd.DataFrame((df[y_km == BC_tmp][df > 0].rank(axis=1) - 1).sum().sort_values(ascending=False)[:10].index, index=rank, columns=['Items'])
  print(f"BC top 10 recommendation for Group {BC_tmp}")
  print(BC)
  print()

#CR
for CR_tmp in [0, 1, 2]:
  df_CR = df[y_km == CR_tmp]

  def calculate_CR(column):
    CR_t = df_CR.sub(column, axis='index')
    CR_one = (CR_t < 0).sum() - (CR_t > 0).sum()
    return CR_one

  CR = df_CR.apply(calculate_CR, axis=0)
  CR[CR < 0] = -1
  CR[CR > 0] = 1
  print(f"CR top 10 recommendation for Group {CR_tmp}")
  print(pd.DataFrame(CR.sum().sort_values(ascending=False)[:10].index, index=rank, columns=['Items']))
  print()
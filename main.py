import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('music_genre.csv', skipinitialspace=True)
df = df.dropna()
tempoindex = df[df.tempo == '?'].index
df.drop(tempoindex, inplace=True)
df.drop(df.loc[df['duration_ms'] == -1.0].index, inplace=True)
df.drop_duplicates(['instance_id','track_name'], inplace=True)
mdf = df.drop(['key','mode','obtained_date'],axis=1)
mdf.to_csv('music_dataset.csv')
reg_dataset = pd.read_csv('music_dataset.csv', index_col=[0])
reg_dataset.head(7)

mdf['music_genre'] = mdf['music_genre'].str.strip('[]').str.replace(' ','').str.replace("'",'')
mdf['music_genre'] = mdf['music_genre'].str.split(',')

# plt.subplots(figsize=(8,6))
# list1 = []
# for i in mdf['music_genre']:
#     list1.extend(i)
# ax = pd.Series(list1).value_counts()[:9].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',9))
# for i, v in enumerate(pd.Series(list1).value_counts()[:9].sort_values(ascending=True).values):
#     ax.text(.6, i, v,fontsize=12,color='white',weight='bold')
# plt.title('Top Genres')
# plt.show()

new_df = reg_dataset.drop(['artist_name','instance_id'],axis=1)
new_df.to_csv('new_music_dataset.csv')
ndf = pd.read_csv('new_music_dataset.csv', index_col=[0])
ndf.head(7)

new_ndf = ndf.drop('track_name',axis="columns")
new_ndf.to_csv('ndf_music_dataset.csv')
num_dataset = pd.read_csv('ndf_music_dataset.csv', index_col=[0])
num_dataset.head(7)

df_uniques = pd.DataFrame([[i, len(num_dataset[i].unique())] for i in num_dataset.columns], columns=['Variable', 'Unique Values']).set_index('Variable')
categorical_variables = list(df_uniques[(10 == df_uniques['Unique Values'])].index)
categorical_variables = list(set(categorical_variables))
numeric_variables = list(set(num_dataset.columns) - set(categorical_variables))


lb, le = LabelBinarizer(), LabelEncoder()

num_dataset = pd.get_dummies(num_dataset, columns = categorical_variables, drop_first=True)

df_join = pd.concat([num_dataset, ndf], axis=1)
df_join = df_join.loc[:,~df_join.T.duplicated(keep='first')]
final_df = df_join.drop('music_genre',axis="columns")
final_df.to_csv('final_music_dataset.csv')
final_df = pd.read_csv('final_music_dataset.csv', index_col=[0])
final_df.head(7)

mm = MinMaxScaler()
for column in [numeric_variables]:
    num_dataset[column] = mm.fit_transform(num_dataset[column])

y, X = num_dataset['music_genre_Rock'], num_dataset.drop(columns='music_genre_Rock')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Estimate KNN model and report outcomes
knn = KNeighborsClassifier(n_neighbors=17)
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Precision, recall, f-score from the multi-class support function
print(classification_report(y_test, y_pred))
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
print('F1 Score: ', round(f1_score(y_test, y_pred), 2))

max_k = 70
f1_scores = list()
error_rates = list()  # 1-accuracy

# keep initialiting the KNN and look at the F1 score and look at the one that maximises it
for k in range(1, max_k):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn = knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    f1 = f1_score(y_pred, y_test)
    f1_scores.append((k, round(f1_score(y_test, y_pred), 4)))
    error = 1 - round(accuracy_score(y_test, y_pred), 4)
    error_rates.append((k, error))

f1_results = pd.DataFrame(f1_scores, columns=['K', 'F1 Score'])
error_results = pd.DataFrame(error_rates, columns=['K', 'Error Rate'])

# Plot F1 results
sns.set_context('talk')
sns.set_style('ticks')

plt.figure(dpi=300)
ax = f1_results.set_index('K').plot(figsize=(14, 9), linewidth=2)
ax.set(xlabel='K', ylabel='f1_scores')
ax.set_xticks(range(1, max_k, 2));
plt.title('KNN F1 Score')
plt.show()

def get_song(song_name):
    song_name = final_df[final_df['track_name']==song_name]
    return song_name

def recommend_song(song_name):
    song = get_song(song_name)
    song_drop = song.drop('track_name', axis=1)
    song_name = pd.DataFrame(data=song_drop, index=None)
    feature_cols = final_df.drop('track_name', axis=1)
    new_X = feature_cols
    neigh = NearestNeighbors(n_neighbors=10, algorithm='kd_tree')
    neigh.fit(new_X)
    distances, indices = neigh.kneighbors(song_name)
    for i in range(len(distances.flatten())):
        return final_df['track_name'].iloc[indices[i]+1].values

print(recommend_song("Candy Shop"))
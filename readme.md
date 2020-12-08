# Time Series Correlation Constellations

Tools to generate 2x2 constellations from time-series correlation plots

## Dependencies

```
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
import math
from matplotlib.patches import Circle
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
```

## Dataset

Dataset is google search trends extracted over 1 Jan to 14 March 2020. 

```
dataset = pd.read_csv(r"dataset.csv")
dataset
```

![Alt text](https://github.com/docligot/constellation_tools/blob/main/dataset.png)

## Plotting the time-series

Visual comparison of time-series. 

```
plt.figure(figsize=(15,4))
plt.plot(dataset[["COVID", "human+rights", "karapatang+pantao"]])
plt.title("Google Search Trends")
plt.xlabel("Date")
plt.ylabel("Search Relevance")
plt.legend(["COVID", "human+rights", "karapatang+pantao"])
plt.show()
```
![Alt text](https://github.com/docligot/constellation_tools/blob/main/time_series.png)

## Generate the correlation table

Correlation table from time-series values. (Note: no all-zero or null columns should be present)

```
dataset_correlation = dataset.corr()
dataset_correlation.to_csv('correlation_table.csv')
dataset_correlation
```
![Alt text](https://github.com/docligot/constellation_tools/blob/main/correlation_table.png)

Heatmap of correlation values (using Excel):

![Alt text](https://github.com/docligot/constellation_tools/blob/main/correlation_heatmap.png)

Converting correlations into distances

```
def rescale_corr(x):
    return 2 - x

dataset_rescale = dataset_correlation.apply(rescale_corr)
dataset_rescale
```

![Alt text](https://github.com/docligot/constellation_tools/blob/main/rescaled_correlation.png)


## Multi-dimensional Scaling

MDS flattens the correlation values into x-y coordinates (or x-y-z if needed)

```
md = MDS(n_components = 2, n_init = 1, dissimilarity = 'precomputed')

def create_mds(x):
    df1 = md.fit_transform(x)
    df2 = pd.DataFrame(df1, columns = ['x_axis', 'y_axis'])
    df3 = pd.DataFrame(x.index, columns=['label'])
    df4 = df2.join(df3, how='inner')
    return df4
```

## Generating MDS Coordinates

```
dataset_mds = create_mds(dataset_rescale)
dataset_mds
```
![Alt text](https://github.com/docligot/constellation_tools/blob/main/mds_coordinates.png)

## Plotting MDS Coordinates

X-y mapping the converted dataset shows a map of the various time-series as they relate to all other values. 
```
def mds_map(x):
    fig = plt.gcf()
    fig.set_size_inches(16, 10.5)
    plt.scatter(x.x_axis, x.y_axis, 150)
    for i, txt in enumerate(x.label):
        plt.annotate(txt, (x.x_axis[i], x.y_axis[i]))

dataset_map = mds_map(dataset_mds)
dataset_map
```
![Alt text](https://github.com/docligot/constellation_tools/blob/main/dataset_map.png)

## KMeans Clustering

Setting up the clustering score: 
```
def getscore(x, y):
    km = KMeans(n_clusters=x)
    z = y.drop(['label'], axis=1)
    km.fit(z)
    return 'Score for ' + str(x) + ' clusters: ' + str(silhouette_score(km.fit_transform(z), km.labels_))

def getscores(x):
    response = []
    for i in range(2,11):
        response.append(getscore(i, x))
    return pd.DataFrame(response, columns=['outcomes'])
```

Testing various clusters for score:

```
dataset_mds.drop(["label"], axis=1)
getscores(dataset_mds)
```
![Alt text](https://github.com/docligot/constellation_tools/blob/main/cluster_score.png)

Get cluster membership (assuming 6 clusters). 

```
km = KMeans(n_clusters = 6)
km.fit(dataset_mds.drop(["label"], axis=1))
clusters = pd.DataFrame(km.predict(dataset_mds.drop(["label"], axis=1)))
clusters.columns = ["cluster"]
dataset_clustered = dataset_mds.join(clusters)
dataset_clustered.sort_values(by='cluster')
```
![Alt text](https://github.com/docligot/constellation_tools/blob/main/cluster_membership.png)

## Drawing the Constellations

```
def draw_galaxy(x, y, x1, x2, y1, y2):
    z = x.drop(['label'], axis = 1)
    fig = plt.gcf()
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 18)
	plt.style.use('dark_background')
    ax.set_facecolor('black')
    km = KMeans(n_clusters=y)
    km.fit(z)
    toutput = km.predict(z)
    toutput1 = pd.DataFrame(toutput, columns = ['cluster'])
    toutput2 = x.join(toutput1, how='inner')
    tds0 = toutput2[toutput2.cluster==0]
    tds1 = toutput2[toutput2.cluster==1]
    tds2 = toutput2[toutput2.cluster==2]
    tds3 = toutput2[toutput2.cluster==3]
    tds4 = toutput2[toutput2.cluster==4]
    tds5 = toutput2[toutput2.cluster==5]
    tds6 = toutput2[toutput2.cluster==6]
    tds7 = toutput2[toutput2.cluster==7]
    tdsc = pd.DataFrame(km.cluster_centers_, columns = ['x_axis', 'y_axis'])
    plt.scatter(tds0.x_axis, tds0.y_axis, 150)
    plt.scatter(tds1.x_axis, tds1.y_axis, 150)
    plt.scatter(tds2.x_axis, tds2.y_axis, 150)
    plt.scatter(tds3.x_axis, tds3.y_axis, 150)
    plt.scatter(tds4.x_axis, tds4.y_axis, 150)
    plt.scatter(tds5.x_axis, tds5.y_axis, 150)
    plt.scatter(tds6.x_axis, tds6.y_axis, 150)
    plt.scatter(tds7.x_axis, tds7.y_axis, 150)
    for i, txt in enumerate(x.label):
        plt.annotate(txt, (x.x_axis[i], x.y_axis[i]))
    plt.axis([x1, x2, y1, y2])
```

Draw the galaxy using 9 clusters
``` 
draw_galaxy(dataset_mds, 9, -2, 2, -2, 2)
```

![Alt text](https://github.com/docligot/constellation_tools/blob/main/galaxy.png)
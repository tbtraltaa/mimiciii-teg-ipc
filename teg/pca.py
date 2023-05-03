import math
import numpy as np      
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_PCA(n_comp, PC, admissions, events, etypes, conf, temporal=True, labels = None, log=False, title=None):
    print("Number of components", n_comp)
    if temporal:
        types = etypes.copy()
        for i in range(int(conf['max_hours']/24)):
            etypes += [t + ' Day ' + str(i + 1) for t in types] 
    n = len(list(admissions.keys()))
    m = len(etypes)
    df = pd.DataFrame(np.zeros((n, m)), index=list(admissions.keys()), columns=etypes)
    loc_count = dict()
    for i, v in PC.items():
        idx = events[i]['id']
        if temporal:
            col = events[i]['type'] + ' Day ' + str(events[i]['t'].days + 1) 
        else:
            col = events[i]['type'] 
        if col in etypes:
            if (idx, col) not in loc_count:
                loc_count[(idx, col)] = 1
            else:
                loc_count[(idx, col)] += 1
        if log:
            df.loc[idx, col] += math.log(v)
        else:
            df.loc[idx, col] += v
    for loc, count in loc_count.items():
        df.loc[loc[0], loc[1]] /= count
    X = df.values
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled= scaler.transform(X)
    pca = PCA(n_components=min(n, m), svd_solver = 'full')
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    print("PCA components")
    print(pca.components_)
    print(pca.explained_variance_ratio_ * 100)
    print(f'Variance explained by {n_comp} principal components', \
            sum(pca.explained_variance_ratio_ * 100))

    print(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance')
    plt.title(title)
    plt.savefig(title)

    pca_3 = PCA(n_components=3, svd_solver = 'full')
    pca_3.fit(X_scaled)
    X_pca_3 = pca_3.transform(X_scaled)
    ax = plt.axes(projection='3d')
    if not labels:
        ax.scatter3D(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2])
    else:
        ax.scatter3D(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2],
                     c=labels)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.title(title + "_3D")
    plt.savefig(title +"_3D")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, 
                             calinski_harabaz_score, silhouette_score)
from scipy.spatial.distance import euclidean, cityblock
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

def intra_to_inter(X, y, dist, r):
    """Compute intracluster to intercluster distance ratio

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    r : integer
        Number of pairs to sample

    Returns
    -------
    ratio : float
        Intracluster to intercluster distance ratio
    """
    p_list = []
    q_list = []
    z = np.random.choice(X.shape[0], size=(r, 2))
    for i, j in z:
        if i == j:
            continue
        elif y[i] == y[j]:
            p_list.append(dist(X[i], X[j]))
        else:
            q_list.append(dist(X[i], X[j]))
    return (np.average(p_list) / np.average(q_list))

def cluster_range(X, clusterer, k_start, k_stop):
    """Calculate the internal validation criteria for different number of
    clusters
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    clusterer : class
        Contains the random state for replicability
    k_start, k_stop : int
        Starting and last number of clusters to perform the internal 
        validation on.

    Returns
    -------
    cluster : dict
        dictionary containing the internal validation values for different k's
    """
    # YOUR CODE HERE
    ys = []
    inertias = []
    chs = []
    scs = []
    iidrs=[]
    
    for k in range(k_start, k_stop+1):
        print('Executing k=', k)
        kmeans_X = KMeans(n_clusters=k, random_state=clusterer.random_state);
        y_predict_X = kmeans_X.fit_predict(X)
        ys.append(y_predict_X)
        inertias.append(kmeans_X.inertia_);
        iidrs.append(intra_to_inter(X, y_predict_X, euclidean, 50))
        chs.append(calinski_harabaz_score(X, y_predict_X));
        scs.append(silhouette_score(X,  y_predict_X));            
        clear_output()
    cluster = {}
    cluster['ys'] = ys
    cluster['inertias'] = inertias
    cluster['chs'] = chs
    cluster['iidrs'] = iidrs
    cluster['scs'] = scs
    return cluster

def plot_internal(kmeans_cluster, k_start, k_stop):
    """Plot the internal validation results
    """
    fig = plt.figure(figsize=(16,8))

    gs = gridspec.GridSpec(4, 2, hspace=0.4)
    ax0 = plt.subplot(gs[:, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[2, 0])
    ax4 = plt.subplot(gs[3, 0])
    ax1.plot(range(k_start, k_stop+1), kmeans_cluster['inertias'], 'o--')
    ax1.set_title('inertias');
    ax1.grid(which='both')
    ax2.plot(range(k_start, k_stop+1), kmeans_cluster['iidrs'], 'o--')
    ax2.set_title('intra-to-inter');
    ax2.grid(which='both')
    ax3.plot(range(k_start, k_stop+1), kmeans_cluster['chs'], 'o--')
    ax3.set_title('calinski-harabaz score');
    ax3.grid(which='both')
    ax4.plot(range(k_start, k_stop+1), kmeans_cluster['scs'], 'o--')
    ax4.set_xlabel('k')
    ax4.set_title('silhouette score');
    ax4.grid(which='both')
    sizes = []
    for i in range(k_stop-k_start+1):
        sizes.append(list(np.unique(kmeans_cluster['ys'][i], return_counts=True)[1]))
    colors = cm.Set3(range(len(sizes[-1])))
    for i in range(len(sizes)):
        j_prev = 0
        
        n = 0
        for j in sizes[i]:
            ax0.bar(i+k_start, j, width=0.6, bottom=j_prev, color=colors[n])
            j_prev += j
            n+=1
    ax0.set_title('cluster sizes');
    ax0.set_xlabel('k');
    return gs

def PC_analysis(X3_data):
    """Plot the variance and cumulative variance explained for different 
    number of features.
    """
    V = np.cov(X3_data, rowvar=False)
    lambdas, w = np.linalg.eig(V)
    indices = np.argsort(lambdas)[::-1]
    lambdas = lambdas[indices]
    w = w[:, indices]
    var_explained = lambdas / lambdas.sum()
    fig, ax = plt.subplots(1, 2, figsize=(16,5))
    ax[0].plot(np.arange(1, len(var_explained)+1), var_explained, lw=3)
    ax[0].set_xlabel('Number of features', size=14)
    ax[0].set_ylabel('Variance explained', size=14);

    cum_var_explained = var_explained.cumsum()
    # fig, ax = plt.subplots(figsize=(12,6))
    n_PC = np.interp(0.8, cum_var_explained.astype(float), np.arange(1,
                                                       len(var_explained)+1))
    ax[1].plot(np.arange(1, len(cum_var_explained)+1), 
               cum_var_explained, '-', zorder=1, lw=3)
    ax[1].set_ylim(ymin=0)
    #ax[1].plot([-50, n_PC], [0.8, 0.8], 'r--')
    ax[1].plot([n_PC, n_PC], [0, 0.8], 'r--')
    ax[1].scatter(n_PC, 0.8, c='r', zorder=2)
    #ax[1].set_xlim(-50,)
    ax[1].set_xlabel('Features', size=14)
    ax[1].set_ylabel('Cumulative variance explained', size=14);

    rate = (var_explained[:-1] - var_explained[1:])/var_explained[:-1]
    for m in range(len(rate)):
        if rate[m] < 0.001:
            break
    m += 1
    ax[0].axvline(m, c='r');
    return ax, n_PC
from pca import *
from sklearn.decomposition import PCA
import numpy as np
    
def are_vectors_similar(v1, v2):
    return np.allclose(v1, v2) or np.allclose(v1, -v2)
    
def test_pca():
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]) 
    n_components = 2
    pca = PCA(n_components)
    
    x1 = pca.fit_transform(X)
    x2 = perform_pca(X, n_components)
    
    for i in range(len(x1)):
        assert are_vectors_similar(x1, x2)
    

    

    



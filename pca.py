import numpy as np

def standardise(feature_matrix):
    # Dimensions of feature matrix.
    n = feature_matrix.shape[0] 
    m = feature_matrix.shape[1]
    
    # Initialise result matrix.
    scaled_feature_matrix = np.zeros((n, m))
    
    # Shifting data so mean = 0.
    for i, col in enumerate(feature_matrix.T):
        col = (col - np.mean(col)) 
        scaled_feature_matrix[:, i] = col 
        
    return scaled_feature_matrix

        
def compute_cov_matrix(feature_matrix):
    # Dimensions of feature matrix.
    n = feature_matrix.shape[0]
    m = feature_matrix.shape[1] 
    
    cov_matrix = np.zeros((m, m))
    feature_matrix_T = feature_matrix.T
    
    for i in range(m):
        for j in range(i, m):
            cov = np.sum(feature_matrix_T[i] * (feature_matrix_T[j])) / (n - 1)
            cov_matrix[i, j] = cov_matrix[j, i] = cov # Covariance matrix is symmetric.
            
    return cov_matrix

    
def compute_eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1] # Order eigenvectors based on eigenvalues.
    
    return eigenvalues[idx], eigenvectors[:, idx] 


def project_onto_principal_components(data_matrix, eigenvectors, n):
    return data_matrix @ eigenvectors[:, :n]
    

def perform_pca(data_matrix, n_components):
    std_matrix = standardise(data_matrix)
    cov_matrix = compute_cov_matrix(std_matrix)
    _, eig_vectors = compute_eigen_decomposition(cov_matrix)
    projected_data = project_onto_principal_components(std_matrix, eig_vectors, n_components)
    
    return projected_data








        
    
        
        

    





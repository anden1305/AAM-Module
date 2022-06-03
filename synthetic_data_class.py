import numpy as np
import pickle

class _synthetic_data:
    
    def __init__(self, N, M, K, p):
        self.X, self.Z, self.A = self.X(N, M, K, p)
        self.N = N
        self.M = M
        self.K = K
        self.p = p
        self.columns = ["SQ "+str(i) for i in range(1,self.M+1)]


    def Z(self, M, K, p):
        # Assure reproducibility
        np.random.seed(42)
        
        betas = np.arange(1, p)
        alphas = np.empty(len(betas))
        Z = np.empty((M, K))
        
        # Calculate beta-values
        betas = betas/sum(betas)
        
        # Calculate alpha-values
        for i in range(len(betas)):
            if not i == (len(betas)-1) or 0:
                alphas[i] = (betas[i]+betas[i+1])/2
            elif i == 0:
                alphas[i] = (0+betas[i])/2
            elif i == (len(betas)-1):
                alphas[i] = (betas[i]+1)/2
        
        # Draw samples from the alphas to construct Z
        
        for i in range(M):
            for j in range(K):
                Z[i,j] = np.random.choice(alphas, size=1)
        
        return Z, betas
    
    
    def A(self, N, K):
        np.random.seed(42) # set another seed :)
        
        alpha = np.ones(K)
        
        return np.random.dirichlet(alpha, size=N).transpose()
    
    def map_X_noise_free(self, X, betas):
        """
        Implement version with noise only..
        """
        M, N = X.shape
        X_thilde = np.empty((M,N))
        
        for i in range(M):
            for j in range(N):
                for k in range(len(betas)):
                    if not k == len(betas)-1:
                        if betas[k] <= X[i,j] and X[i,j] <= betas[k+1]:
                            X_thilde[i,j] = int(k+1)
                            
                        elif X[i,j] < betas[0]:
                            X_thilde[i,j] = int(1)
                        
                        elif X[i,j] > betas[-1]:
                            X_thilde[i,j] = int(len(betas))
        X_thilde = X_thilde.astype(int)
        return X_thilde    
    
    def X(self, N, M, K, p):
        
        Z, betas = self.Z(M=M, K=K, p=p)
        A = self.A(N=N, K=K)
        X_hat = Z@A
        
        X_thilde = self.map_X_noise_free(X=X_hat, betas=betas)
        
        return X_thilde, Z, A

    def _save(self,type,filename):
        file = open("synthetic_results/" + type + "_" + filename + '_metadata' + '.obj','wb')
        pickle.dump(self, file)
        file.close()
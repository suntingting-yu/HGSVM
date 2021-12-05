import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy.spatial.distance import cdist
import itertools
from sklearn.cluster import KMeans
class HyperSVM(object):
    def __init__(self, opt):
        self.opt = opt

    def dist2(self, x, c):
        print("----------------------------------------------------------------")
        print('Calculate squared distance between two sets of points')
        (rowx, colx) = x.shape
        (rowc, colc) = c.shape
        if colc != colx:
            print('Data dimension does not match dimension of centres')
            return
        x2T = sparse.csc_matrix(np.multiply(x, x)).T
        c2T = sparse.csc_matrix(np.multiply(c, c)).T
        a2 = (np.dot(np.ones((rowc, 1)), x2T.sum(axis=0))).T
        b2 = np.dot(np.ones((rowx, 1)), c2T.sum(axis=0))
        ab2 = 2 * np.dot(x, c.T)
        distance = a2 + b2 - ab2
        distance = np.array(distance)
        return distance

    # calculate Adjacency matrix
    def Adjacency(self, X):
        # X = np.vstack([X, X_u])
        if self.opt['neighbor_mode'] == 'connectivity':
            W = kneighbors_graph(X, self.opt['n_neighbor'], mode='connectivity', include_self=False)
            W = W.A
        elif self.opt['neighbor_mode'] == 'distance':
            W = kneighbors_graph(X, self.opt['n_neighbor'], mode='distance', include_self=False)
            W = W.maximum(W.T)
            W = sparse.csr_matrix((np.exp(-W.data ** 2 / 4 / self.opt['t']), W.indices, W.indptr),
                                  shape=(X.shape[0], X.shape[0]))
        else:
            raise Exception()
        return W

    def GetRawWeightByAffinity(self, AMatrix, AttMat):
        # Generating the raw weights of hyperedges via using the affinity matrix  
        print('------------------------------------------------------------------------')
        print('Generating the raw weights of hyperedges via using the affinity matrix')
        (row, col) = AttMat.shape
        W = np.zeros(col)
        for i in range(col):
            ind = np.where(AttMat[:, i] != 0)
            length = len(ind[0])
            if length != 1 and length != 0:
                sum = 0
                for m in ind[0]:
                    for n in ind[0]:
                        sum += np.exp(-AMatrix[m, n])
                W[i] = sum / (length * (length - 1))
            else:
                W[i] = np.inf
        return W

    # the vertex-edge incident matrix is exactly the attribute label matrix
    def Getverx_edgeIncident(self, X):
        self.X = X
        print('------------------------------------------------------------------------')
        print('Generating the the vertex-edge incident matrix')
        A = self.Adjacency(self.X)
        N = len(A)
        need = []
        row = []
        EDGE = np.zeros((1, N + 1))
        bc = []
        for i in range(1, N + 1):
            seed = A[i - 1, :]
            mem = np.nonzero(seed == 1)[0] + 1
            k = len(mem)
            EDGEtemp = np.zeros((1, N + 1))
            EDGETT = np.zeros((1, N + 1))
            p = 0;
            sumx = sum(np.in1d(mem, EDGEtemp))
            while sum(np.in1d(mem, EDGEtemp)) != len(mem):
                C = np.array(list(itertools.combinations(mem, k)))
                a, b = np.shape(C)
                for j in range(a):
                    row = C[j, :]
                    x = np.in1d(mem, EDGEtemp)
                    y = np.where(x == 0)
                    need = mem[y]
                    if sum(np.in1d(need, row)) != 0:
                        G = []
                        for ri in range(0, len(row)):
                            rx = row[ri]
                            for rj in range(0, len(row)):
                                ry = row[rj]
                                bc = np.append(bc, A[rx - 1][ry - 1])
                                # bc.append(A[rx - 1][ry - 1])
                            # G.append(bc)
                            YYY = len(G)
                            if len(G) == 0:
                                G = np.append(G, bc, axis=0)
                            else:
                                G = np.vstack((G, bc))
                            bc = []
                        Gn = len(G)
                        Gsum = np.sum(G)
                        if Gsum == (Gn ** 2 - Gn):
                            Irow = np.hstack((i, row))
                            index = 0
                            if p > 0:
                                EDGEtemp = np.vstack((EDGEtemp, EDGETT))
                                while index < len(Irow):
                                    EDGEtemp[p, index] = Irow[index]
                                    index = index + 1
                            else:
                                while index < len(Irow):
                                    EDGEtemp[p, index] = Irow[index]
                                    index = index + 1
                            p = p + 1
                k = k - 1
            EDGE = np.vstack((EDGE, EDGEtemp))
        # X=np.sort(EDGE,axis=1)
        # un, un_i, v = np.unique(X.view(X.dtype.descr * X.shape[1]),return_index=True,return_inverse=True)
        un, un_i, v = np.unique(
            np.sort(EDGE, axis=1).view(np.sort(EDGE, axis=1).dtype.descr * np.sort(EDGE, axis=1).shape[1]),
            return_index=True, return_inverse=True)
        X = np.array([np.sort(un_i)]).T
        Hgraph = []
        for hi in range(0, len(X)):
            ll = EDGE[X[hi], :]
            if len(Hgraph) == 0:
                Hgraph = np.append(Hgraph, ll)
            else:
                Hgraph = np.vstack((Hgraph, ll))
        Hgraph = Hgraph[1:len(X)]

        [a, b] = np.shape(Hgraph)
        H = np.zeros((a, N))
        for i in range(0, a):
            row = Hgraph[i, :]
            d = np.where(row == 0)[0]
            for di in d:
                row[di] = 0
            for rri in row:
                if rri != 0:
                    H[i, np.int64(rri) - 1] = 1
        H = H.T
        return H

    def ConstructHyperLaplacian(self, W, H, mu):
        # Constructing the Hypergraph Laplacian matrix
        print('------------------------------------------------------------------------')
        print('Constructing the Hypergraph Laplacian matrix')
        inds = np.where(W != np.inf)
        W = W[inds[0]]
        H = H[:, inds[0]]
        # normalize the raw weights and kernelize them to present the final weights
        # temp = scipy.mean(W) * mu
        # W = scipy.exp(-W / (scipy.mean(W) * mu))
        # the diagonal weight matrix
        W = np.diag(W)
        # the diagonal vertex degree matrix
        Ve = np.diag(H.sum(axis=0))

        # zhou's method for constructing the normalized Laplacian of a hypergraph
        Ve = np.diag(1 / np.diag(Ve))

        Dv = ((np.dot(H, W)).T).sum(axis=0)
        NewW = np.dot(np.dot(np.dot(H, W), Ve), H.T)
        Dv_I = 1 / Dv
        Dv_I = np.diag(Dv_I)
        Lap = np.dot(np.dot(np.sqrt(Dv_I), (np.diag(Dv) - NewW)), np.sqrt(Dv_I))
        return Lap

    def fit(self, X, Y, X_u):
        # construct graph
        self.X = np.vstack([X, X_u])
        Y = np.diag(Y)
        distance = self.dist2(self.X, self.X)
        # Raw Weights of hyperedge, each attribute is corresponding to a hyperedge
        print(distance)
        # the vertex-edge incident matrix is exactly the attribute label matrix
        H = self.Getverx_edgeIncident(self.X)
        print(H)
        # Raw Weights of hyperedge, each attribute is corresponding to a hyperedge
        W = self.GetRawWeightByAffinity(distance, H)
        L = self.ConstructHyperLaplacian(W, H, 1)
        print(L)
        # Computing K with k(i,j) = kernel(i, j)
        K = self.opt['kernel_function'](self.X, self.X, **self.opt['kernel_parameters'])
        l = X.shape[0]
        u = X_u.shape[0]
        # Creating matrix J [I (l x l), 0 (l x (l+u))]
        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)
        # Computing "almost" alpha
        almost_alpha = np.linalg.inv(2 * self.opt['gamma_A'] * np.identity(l + u) \
                                     + ((2 * self.opt['gamma_I']) / (l + u) ** 2) * L.dot(K)).dot(J.T).dot(Y)
        # Computing Q
        Q = Y.dot(J).dot(K).dot(almost_alpha)
        Q = (Q + Q.T) / 2

        del W, L, K, J

        e = np.ones(l)
        q = -e

        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)

        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(Q) + q))

        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / l) for _ in range(l)]

        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(np.diag(Y))

        def constraint_grad(beta):
            return np.diag(Y)

        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}

        # ===== Solving =====
        x0 = np.zeros(l)

        beta_hat = minimize(objective_func, x0, jac=objective_grad, constraints=cons, bounds=bounds)['x']

        # Computing final alpha
        self.alpha = almost_alpha.dot(beta_hat)

        del almost_alpha, Q

        # Finding optimal decision boundary b using labeled data
        new_K = self.opt['kernel_function'](self.X, X, **self.opt['kernel_parameters'])
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        self.sv_ind = np.nonzero((beta_hat >= 1e-7) * (beta_hat <= (1 / l) - 1e-7))[0]
        ind = self.sv_ind[0]
        self.b = np.diag(Y)[ind] - f[ind]

    def decision_function(self, X):
        new_K = self.opt['kernel_function'](self.X, X, **self.opt['kernel_parameters'])
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f + self.b
def rbf(X1, X2, **kwargs):
    return np.exp(-cdist(X1, X2) ** 2 * kwargs['gamma'])




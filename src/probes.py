
import torch as t
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, SparsePCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC


class MMProbe(t.nn.Module):
    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        if inv is None:
            self.inv = t.nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
        else:
            self.inv = t.nn.Parameter(inv, requires_grad=False)

    def forward(self, x, iid=False):
        if not (t.is_tensor(x)):
             x = t.tensor(x) 
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            return t.nn.Sigmoid()(x @ self.direction)

    def pred(self, x, iid=False):
        if not (t.is_tensor(x)):
             x = t.tensor(x) 
        return self(x, iid=iid).round()

    def from_data(acts, labels, atol=1e-3, device='cpu'):
        if not (t.is_tensor(acts) and t.is_tensor(labels)):
             labels = t.tensor(labels) 
             acts = t.tensor(acts)
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]
        
        probe = MMProbe(direction, covariance=covariance).to(device)

        return probe

    def fit(self, x, y):
        return self.from_data()
        
    def predict(self, x):
        return self.pred(t.tensor(np.array(x)).double())


class ProjectionModel:
    def __init__(self, conf):
        self.model_type = self.__class__.__name__
        self.score = None
        self.model = None
        self.cfg = conf
    def train(self, train_data):
        raise NotImplementedError("train method must be implemented in the subclass")
    def project(self, data):
        raise NotImplementedError("project method must be implemented in the subclass")
    def inverse(self, data):
        raise NotImplementedError("project method must be implemented in the subclass")
    def data_to_numpy(self, some_data, labels=False):
        X = np.array([data_elt.activations for data_elt in some_data])
        if labels:
            Y = np.array([data_elt.label for data_elt in some_data])
            return X, Y
        return X
    def fwd(self, data):
        return self.model.transform(np.array([data.activations]))[0]

class PCAProjectionModel(ProjectionModel):
    """
        PCA Projection
    """
    def train(self, train_data):
        X = self.data_to_numpy(train_data)
        self.model = PCA(**self.cfg["projections"]["PCAProjectionModel"])
        self.model.fit(X)
        # self.score = self.model.get_precision()

    def project(self, data, raw=False):
        if raw:
            return self.model.transform(data)
        return self.model.transform(self.data_to_numpy(data))

    def inverse(self, data):
        return self.model.inverse_transform(data)
    
class NoProjectionModel(ProjectionModel):
    """
        No Projection
    """
    def train(self, train_data):
        pass

    def project(self, data, raw=False):
        if raw:
            return data
        return self.data_to_numpy(data)

    def inverse(self, data):
        return data

    def fwd(self, data):
        return np.array([data.activations])[0]
    
class LDAProjectionModel(ProjectionModel):
    """
        LDA Projection
    """
    def train(self, train_data):
        X, Y = self.data_to_numpy(train_data, labels=True)
        self.model = LDA()
        self.model.fit(X, Y)
        self.score = self.model.score(X, Y)
        self.calculate_pseudo_inverse()

    def calculate_pseudo_inverse(self):
        # Adjust W to match the number of projected dimensions
        W = self.model.scalings_[:, :1]
        # Compute the pseudo-inverse of the transformation matrix
        self.W_pseudo_inv = np.linalg.pinv(W)

    def project(self, data, raw=False):
        if raw:
            return self.model.transform(data)
        return self.model.transform(self.data_to_numpy(data))

    # def inverse(self, data):
    #     return self.model.inverse_transform(data)
    
    def inverse(self, X_projected):

        X_projected = np.array(X_projected)
        # if not hasattr(self, 'W_pseudo_inv'):
        #     self.calculate_pseudo_inverse()

        # Adjust W to match the number of projected dimensions
        W = self.model.scalings_[:, :1]

        X_projected = X_projected[:, np.newaxis]

        # Compute the pseudo-inverse of the transformation matrix
        self.W_pseudo_inv = np.linalg.pinv(W)

        # Add back the mean of the original data
        X_approx = np.dot(X_projected, W.T) + self.model.xbar_

        return X_approx[0]

    
class SparsePCAProjectionModel(ProjectionModel):
    """
        Sparse PCA Projection
    """
    def train(self, train_data):
        X = self.data_to_numpy(train_data)
        self.model = SparsePCA(**self.cfg["projections"]["SparsePCAProjectionModel"])
        self.model.fit(X)
        print("Non zero components in Sparse PCA:", self.non_zero_components())

    def non_zero_components(self):
        # Count the number of non-zero elements in each component
        return np.count_nonzero(self.model.components_, axis=1)

    def project(self, data, raw=False):
        if raw:
            return self.model.transform(data)
        return self.model.transform(self.data_to_numpy(data))

    def inverse(self, data):
        return self.model.inverse_transform(data)

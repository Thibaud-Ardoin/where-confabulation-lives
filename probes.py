
import torch as t

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

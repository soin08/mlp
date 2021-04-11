import numpy as np
import sys

import numpy as np

from copy import deepcopy


class Linear:
    def __init__(self, n_input, n_output, seed=None):
        random = np.random.RandomState(seed)
        self.b = np.zeros(n_output)
        self.w = random.normal(loc=0.0, scale=0.1,
                                size=(n_input, n_output))
        self.z = None
        self.a = None

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        self.z = np.dot(X, self.w) + self.b

        # step 2: activation of hidden layer
        self.a = self._sigmoid(self.z)

        return self.a


class MLP:
    def __init__(self, n_features=2, n_labels=2, hidden=(5, 10, 5), l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.epochs = epochs
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.l2 = l2
        self.eta = eta
        self.layers = []
        self._init_layers(n_features, n_labels, hidden, seed)

    def _init_layers(self, num_features, num_labels, hidden_layers, seed):
        self.layers.append(Linear(num_features, hidden_layers[0], seed))

        i = 0
        for i in range(1, len(hidden_layers)):
            self.layers.append(Linear(hidden_layers[i - 1], hidden_layers[i], seed))

        self.layers.append(Linear(hidden_layers[i], num_labels, seed))

    def _onehot(self, y, n_classes):
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _compute_cost(self, y_enc, output):
        L2_term = self.l2 * sum([np.sum(layer.w ** 2.) for layer in self.layers])
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        y_pred = self._forward(X)
        return np.argmax(y_pred, axis=1)

    def _forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def _backward(self, X, y):
        prev_layer, layer, next_layer, sigma = [None] * 4

        for i in reversed(range(len(self.layers))):
            try:
                layer = self.layers[i]
                prev_layer = self.layers[i - 1]
                next_layer = self.layers[i + 1]
            except IndexError:
                pass

            if i == len(self.layers) - 1:  # output layer
                sigma = layer.a - y
                grad_w = np.dot(prev_layer.a.T, sigma)
            elif i > 0:  # hidden layer
                sigmoid_derivative = layer.a * (1. - layer.a)
                sigma = (np.dot(sigma, next_layer.w.T) * sigmoid_derivative)
                grad_w = np.dot(prev_layer.a.T, sigma)
            else:  # input layer
                sigmoid_derivative = layer.a * (1. - layer.a)
                sigma = (np.dot(sigma, next_layer.w.T) * sigmoid_derivative)
                grad_w = np.dot(X.T, sigma)

            grad_b = np.sum(sigma, axis=0)
            delta_w = (grad_w + self.l2 * layer.w)
            delta_b = grad_b
            layer.w -= self.eta * delta_w
            layer.b -= self.eta * delta_b
            
    def collect_weights(self):
        w = [(layer.b, layer.w, layer.z, layer.a) for layer in self.layers]
        #print("weights:")
        #print(w)
        return w
    
    def set_weights(self, weights):
        for i, w in enumerate(weights):
            #print(f'prev weight a: {self.layers[i].a}, new weight:{w}')
            #print(f'prev weight w: {self.layers[i].w}, new weight:{w[1]}')
            self.layers[i].b = w[0]
            self.layers[i].w = w[1]
            self.layers[i].z = w[2]
            self.layers[i].a = w[3]
            #self.layers[i].w = w[1]

    def fit(self, X_train, y_train, X_valid, y_valid):
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': [], 'saved_weights': [], 'saved_layers': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):
            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):

                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                self._forward(X_train[batch_idx])
                self._backward(X_train[batch_idx], y_train_enc[batch_idx])

            a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
            self.eval_['saved_weights'].append(self.collect_weights())
            self.eval_['saved_layers'].append(deepcopy(self.layers))
            
        max_valid_ix = np.argmax(self.eval_['valid_acc'])
        
        
        # self.set_weights(self.eval_['saved_weights'][max_valid_ix])
        
        self.layers = self.eval_['saved_layers'][max_valid_ix]
        self.eval_['saved_layers'] = []
        
        y_train_pred = self.predict(X_train)
        y_valid_pred = self.predict(X_valid)
        
        train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
        
        valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                     X_valid.shape[0])

        sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                         '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                         (epoch_strlen, i+1, self.epochs, cost,
                          train_acc*100, valid_acc*100))

        
        sys.stderr.flush()

        return self

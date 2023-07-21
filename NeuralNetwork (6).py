import numpy as np
import sys

def read_csv(fn):
  with open(fn) as fd:
    data = np.array([
          [
            float(i) for i in line.split(',')
          ] 
          for line in fd.read().split('\n') if line!='']
        )
  return data

def read_data(tn, tl):
    train_data = read_csv(tn)
    train_labels = read_csv(tl)
    return train_data, train_labels


class CrossEntropy():
    def loss(self, y, y_hat):
        n = len(y)
        lp = - np.log(y_hat[np.arange(n), y.argmax(axis=1)])
        return np.sum(lp)/n
        
    def grad(self, y, y_hat):
        n = y.shape[0]
        res = y_hat - y
        return res/n

class Sigmoid():
  def __init__(self):
    self.name = 'sigmoid'
    
  def act(self, inp):  
    return 1 / (1 + np.exp(-inp))
  
  def grad(self, inp):
    return inp * (1 - inp)

class SoftmaxActivation():
    def __init__(self):
        self.name = 'softmax'
        
    def act(self, inp):
        e = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def prime(self, inp):
        return inp * (1 - inp)
        
activations = {
    'sigmoid': Sigmoid,
    'softmax': SoftmaxActivation
}

class FullyConnectedLayer():
  def __init__(self, input_dims, op_dims, activation):
    self.w = np.random.rand(input_dims, op_dims) 
    self.b = np.zeros((1, op_dims))
    self.act = activations[activation]()
    

  def fp(self, inp):
    self.inp = inp
    self.op = np.dot(self.inp, self.w) + self.b
    self.op = self.act.act(self.op)
    return self.op

  def bp(self, oe, lr):
    if self.act.name != 'softmax':
        oe = self.act.grad(self.op) * oe
    
    ie = np.dot(oe, self.w.T)
    we = np.dot(self.inp.T, oe)
    self.w -= lr * we

    if self.act.name == 'softmax':
        self.b -= lr * np.sum(oe, axis=0, keepdims=True)
    else:
        self.b -= lr * np.sum(oe, axis=0)
    return ie

class Model:
  def __init__(self):
    self._los_ = None
    self.ls = []

  def add(self, layer):
    self.ls.append(layer)

  def loss(self, l):
    self._los_ = l

  def predict(self, inp):
    op = inp
    for layer in self.ls:
        op = layer.fp(op)
    return op
  
  def train(self, x, y, epcs, 
                          lr, bs=200):
    N = len(x)
    history = []
    for ep in range(epcs):
      err = 0
      for j in range(int(N/bs)):
        op = self.predict(
              x[j:j+bs]
            )
        err += self._los_.loss(y[j:j+bs], op)

        __err__ = self._los_.grad(y[j:j+bs], op)
        for l in reversed(self.ls):
            __err__ = l.bp(__err__, lr)

      if ep%100 == 0:
        print("Epoch {}/{}  error={}".format(ep+1, epcs, err))
      if err < 0.03:
        break
      history.append({
          'epoch': ep,
          'err': err
      })
    return history

if __name__ == '__main__':
    args = sys.argv
    tdn = args[1]
    tln = args[2]
    test_data_name = args[3]
    train_x, train_y = read_data(tdn, tln)
    test_x = read_csv(test_data_name)

   
    a = train_y.astype(int).flatten()
    train_y = np.zeros((a.size, a.max() + 1))
    train_y[np.arange(a.size), a] = 1

    net = Model()

    net.add(FullyConnectedLayer(2, 128, 'sigmoid'))
    net.add(FullyConnectedLayer(128, 128, 'sigmoid'))
    net.add(FullyConnectedLayer(128, 2, 'softmax'))
    net.loss(CrossEntropy())

    hist = net.train(train_x, train_y, epcs=8000, lr=0.7)


    preds = net.predict(test_x).argmax(axis=1)
    file = open("test_predictions.csv", "w")
    for i in preds:
      file.write(str(i)+'\n')
    file.close()
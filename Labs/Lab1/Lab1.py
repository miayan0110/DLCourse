import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
  import numpy as np
  pts = np.random.uniform(0, 1, (n, 2))
  inputs = []
  labels = []
  for pt in pts:
    inputs.append([pt[0], pt[1]])
    distance = (pt[0]-pt[1])/1.414
    if pt[0] > pt[1]:
      labels.append(0)
    else:
      labels.append(1)
  return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
  inputs = []
  labels = []
  
  for i in range(11):
    inputs.append([0.1*i, 0.1*i])
    labels.append(0)
  
    if 0.1*i == 0.5:
      continue
  
    inputs.append([0.1*i, 1-0.1*i])
    labels.append(1)
  return np.array(inputs), np.array(labels).reshape(21, 1)

class Layer:
  def __init__(self, this_neurons, next_neurons) -> None:
    def weight_init(this_neurons, next_neurons):
      return np.random.randn(this_neurons, next_neurons)

    self.inputs = np.zeros((next_neurons,))
    self.outputs = np.zeros((next_neurons,))
    self.weights = weight_init(this_neurons, next_neurons)
    self.gradients = np.array([])

class NeuralNetwork:
  def __init__(self, layersDefinition, activation='sigmoid', learningRate=0.1) -> None:
    # initialize
    def layer_init():
      layers = []
      for i in range(len(layersDefinition)-1):
        layers.append(Layer(layersDefinition[i], layersDefinition[i+1]))
      return layers

    self.x = np.array([])
    self.layers = layer_init()
    self.num_of_layers = len(self.layers)
    self.learning_rate = learningRate
    self.epoch = 0
    self.trainLoss = []
    self.trainAccuracy = []
    match activation:
      case 'sigmoid':
        self.activation = self.sigmoid
        self.deriv = self.derivative_sigmoid
      case 'relu':
        self.activation = self.relu
        self.deriv = self.derivative_relu
      case _:
        self.activation = lambda x: x
        self.deriv = lambda x: 1

  # back propagation
  def forward_pass(self, x):
    for i in range(self.num_of_layers):
      if i == 0:
        self.layers[i].inputs = x.dot(self.layers[i].weights)
      else:
        self.layers[i].inputs = self.layers[i-1].outputs.dot(self.layers[i].weights)
      self.layers[i].outputs = self.activation(self.layers[i].inputs)
      # print('layer[{}].outputs = {}'.format(i, self.layers[i].outputs))
    return self.layers[-1].outputs

  def backward_pass(self, error):
    delta = np.array(error * self.deriv(self.layers[-1].outputs))
    for i in range(self.num_of_layers-1, -1, -1):
      if i == self.num_of_layers-1:
        self.layers[i].gradients = delta
      else:
        delta = self.layers[i+1].gradients.dot(self.layers[i+1].weights.T)
        delta = delta * self.deriv(self.layers[i].outputs)
        self.layers[i].gradients = delta

  def update(self):
    for i in range(self.num_of_layers):
      if i == 0:
        self.layers[i].weights -= self.learning_rate * self.x.T.dot(self.layers[i].gradients)
      else:
        self.layers[i].weights -= self.learning_rate * self.layers[i-1].outputs.T.dot(self.layers[i].gradients)

  def train(self , x, yhat, epoch=10):
    self.x = np.array(x)
    self.epoch = epoch
    for i in range(epoch):
      y = self.forward_pass(x)
      l = self.mse(y, yhat)
      error = y-yhat
      self.backward_pass(error)
      self.update()
      print('epoch {:4d}\tloss : {}'.format(i, l))
      # print('epoch {:4d}\ty : {}\tloss : {}'.format(i, y, l))
      acc = self.accuracy(y, yhat)
      self.trainLoss.append(l)
      self.trainAccuracy.append(acc)

  # testing
  def test(self, x, yhat):
    pred_y = self.forward_pass(x)
    l = self.mse(pred_y, yhat)
    for i in range(len(pred_y)):
      print('| Iter{:2d} | Ground truth: {:1.1f} | prediction: {:.5f} |'.format(i, yhat.flatten()[i], pred_y.flatten()[i]))
    acc = self.accuracy(pred_y, yhat)
    print('loss={:.5f} accuracy={:3.2f}%'.format(l, acc*100))
    return pred_y

  # activation function
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))
  def derivative_sigmoid(self, x):
    return np.multiply(x, 1.0-x)
  
  def relu(self, x):
    return x * (x > 0)
  def derivative_relu(self, x):
    return 1 * (x > 0)

  # loss function
  def mse(self, y, yhat):
    return ((y - yhat)**2).mean()

  # accuracy function
  def accuracy(self, pred_y, yhat):
    if len(pred_y) != len(yhat):
      return 'error!'
    for i in range(len(pred_y)):
      pred_y[i] = 1.0 if pred_y[i] > 0.5 else 0.0
    return (pred_y == yhat).sum()/len(pred_y)

  # show result
  def plot(self, attr):
    plt.plot(attr)
    plt.xlim(0, self.epoch)
    plt.show()

  def show_result(self, x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
      if y[i] == 0:
        plt.plot(x[i][0], x[i][1], 'ro')
      else:
        plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
      if pred_y[i] == 0:
        plt.plot(x[i][0], x[i][1], 'ro')
      else:
        plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

if __name__ == '__main__':
  # x, yhat = generate_XOR_easy()
  x, yhat = generate_linear(n=100)

  myNN = NeuralNetwork([2,3,3,1], 'relu', 0.5)
  myNN.train(x, yhat, 2)
  myNN.plot(myNN.trainLoss)
  pred_y = myNN.test(x, yhat)
  myNN.show_result(x, yhat, pred_y)
  
import numpy as np

class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output_grad):
        raise NotImplementedError
        
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.input = None
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, input):
        self.input = input
        output = np.dot(input, self.weight) + self.bias
        return output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = np.dot(self.input.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0, keepdims=True)
        return input_grad

class Activation1(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)  # ReLU activation
        return output        

    def backward(self, output_grad):
        input_grad = output_grad * (self.input > 0)  # Gradient for ReLU
        return input_grad


import numpy as np

class SoftmaxWithLoss(_Layer):
    def __init__(self):
        super().__init__()
        self.predict = None
        self.target = None

    def forward(self, input, target):
        self.target = target

        
        input -= np.max(input, axis=1, keepdims=True)
        exp_input = np.exp(input)
        self.predict = exp_input / np.sum(exp_input, axis=1, keepdims=True)

        
        if self.target.ndim == 2:
            self.target = np.argmax(self.target, axis=1)

        
        m = self.predict.shape[0]
        log_likelihood = -np.log(self.predict[np.arange(m), self.target] + 1e-10)
        your_loss = np.sum(log_likelihood) / m
        
        return self.predict, your_loss

    def backward(self):
        m = self.predict.shape[0]
        
        target_one_hot = np.zeros_like(self.predict)
        target_one_hot[np.arange(m), self.target] = 1
        input_grad = (self.predict - target_one_hot) / m
        return input_grad



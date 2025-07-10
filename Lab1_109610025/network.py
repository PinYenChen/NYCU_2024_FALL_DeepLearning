from model.layer import FullyConnected, Activation1, SoftmaxWithLoss

class Network(object):
    def __init__(self):
        self.fc1 = FullyConnected(28*28, 256)  
        self.act1 = Activation1()             
        self.fc2 = FullyConnected(256, 128)     
        self.act2 = Activation1()             
        self.fc3 = FullyConnected(128, 64)    
        self.act3 = Activation1()             
        self.fc4 = FullyConnected(64, 10)     
        self.loss = SoftmaxWithLoss()         

    def forward(self, input, target):
        h1 = self.fc1.forward(input)
        a1 = self.act1.forward(h1)            
        h2 = self.fc2.forward(a1)
        a2 = self.act2.forward(h2)            
        h3 = self.fc3.forward(a2)
        a3 = self.act3.forward(h3)            
        h4 = self.fc4.forward(a3)             
        pred, loss = self.loss.forward(h4, target)
        return pred, loss

    def backward(self):
        grad_h4 = self.loss.backward()         
        grad_a3 = self.fc4.backward(grad_h4)   
        grad_h3 = self.act3.backward(grad_a3)  
        grad_a2 = self.fc3.backward(grad_h3)   
        grad_h2 = self.act2.backward(grad_a2)  
        grad_a1 = self.fc2.backward(grad_h2)   
        grad_h1 = self.act1.backward(grad_a1)  
        self.fc1.backward(grad_h1)             

    def update(self, lr):
        
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad
        self.fc2.weight -= lr * self.fc2.weight_grad
        self.fc2.bias -= lr * self.fc2.bias_grad
        self.fc3.weight -= lr * self.fc3.weight_grad
        self.fc3.bias -= lr * self.fc3.bias_grad
        self.fc4.weight -= lr * self.fc4.weight_grad
        self.fc4.bias -= lr * self.fc4.bias_grad



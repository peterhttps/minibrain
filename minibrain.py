import numpy as np
from progress.bar import ShadyBar

class NeuralNetwork:
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2, activaction_function="relu"):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs 
        self.activation_function = activaction_function

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # Random weights
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w) 

            if (self.activation_function == 'relu'):
                activations = self.leaky_relu(net_inputs)
            elif (self.activation_function == 'sigmoid'):
                activations = self.sigmoid(net_inputs)
            elif (self.activation_function == 'tanh'):
                activations = self.tanh(net_inputs)
           # print(activations)
            self.activations[i + 1] = activations

        #print(activations)
        return activations
    

    def back_propagate(self, error, verbose=False):
        #print(len(self.derivatives))

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
        
            if (self.activation_function == 'relu'):
                delta = error * self.d_leaky_relu(activations)
            elif (self.activation_function == 'sigmoid'):
                delta = error * self.sigmoid_derivative(activations)
            elif (self.activation_function == 'tanh'):
                delta = error * self.d_tanh(activations)
            

            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    
    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate


    def train(self, inputs, targets, epochs, learning_rate):
        bar = ShadyBar('Training', max=epochs)

        for i in range(epochs):
            sum_error = 0
            for j, (input, target) in enumerate(zip(inputs, targets)):
                output = self.forward_propagate(input)
                error = target - output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)

                sum_error += self.mse(target, output)

            #print("Error: {} at epoch {}".format(sum_error / len(inputs), i))
            bar.next()
        
        print()
        print("Loss: " + str(sum_error / len(inputs)))

    def predict(self, input):
        output = self.forward_propagate(input)
        return output

    def mse(self, target, output):
        return np.average((target - output)**2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(x, 0)

    def d_relu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.001)

    def d_leaky_relu(self, x):
        x[x>=0] = 1
        x[x<0] = 0.001
        return x

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1.0 - np.tanh(x)**2


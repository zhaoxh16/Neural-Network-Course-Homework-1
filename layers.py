import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)
        self.input_now = None

    def forward(self, input):
        '''Your codes here'''
        """
        input: [batch_size, in_num]
        output: [batch_size, in_num]
        """
        self.input_now = input
        return np.maximum(input, 0)

    def backward(self, grad_output):
        '''Your codes here'''
        """
        grad_output: [in_num, batch_size]
        output: [in_num, batch_size]
        """
        input_deriv = (self.input_now > 0)
        return input_deriv.T * grad_output


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)
        self.sigmoid_result = None

    def forward(self, input):
        '''Your codes here'''
        # print("input", self.name, input)
        self.sigmoid_result = 1/(1+np.exp(-input))
        return self.sigmoid_result

    def backward(self, grad_output):
        '''Your codes here'''
        sigmoid_result_deriv = self.sigmoid_result * (1-self.sigmoid_result)
        # print("output", self.name, self.sigmoid_result, grad_output, sigmoid_result_deriv.T * grad_output)
        return sigmoid_result_deriv.T * grad_output


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std=0.01):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

        self.input_now = None

    def forward(self, input):
        '''Your codes here'''
        """
        input: [*, in_num]
        output: [*, out_num]
        """
        self.input_now = input
        return input.dot(self.W) + self.b

    def backward(self, grad_output):
        '''Your codes here'''
        """
        grad_output: [out_num, batch_size]
        output: [in_num, batch_size]
        """
        batch_size = grad_output.shape[1]
        self.grad_W = (grad_output.dot(self.input_now)).T / batch_size
        self.grad_b = np.mean(grad_output, axis=-1)
        return self.W.dot(grad_output)

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b

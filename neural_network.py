import numpy as np
import config as cfg

class NeuralNetwork:
    def __init__(self, input_size=cfg.INPUT_SIZE, hidden_size=cfg.HIDDEN_SIZE, output_size=cfg.OUTPUT_SIZE):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        limit_ih = np.sqrt(6. / (self.input_size + self.hidden_size))
        self.weights_input_hidden = np.random.uniform(-limit_ih, limit_ih, (self.input_size, self.hidden_size))
        self.bias_hidden = np.zeros(self.hidden_size)
        limit_ho = np.sqrt(6. / (self.hidden_size + self.output_size))
        self.weights_hidden_output = np.random.uniform(-limit_ho, limit_ho, (self.hidden_size, self.output_size))
        self.bias_output = np.zeros(self.output_size)

    def forward(self, inputs):
        inputs = np.array(inputs).flatten()
        if inputs.shape[0] != self.input_size:
            inputs = np.resize(inputs, self.input_size)
        hidden_raw = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_activated = self.tanh(hidden_raw)
        output_raw = np.dot(hidden_activated, self.weights_hidden_output) + self.bias_output
        output_activated = self.sigmoid(output_raw)
        return output_activated[0]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -15, 15))

    def clone(self):
        clone = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        clone.weights_input_hidden = np.copy(self.weights_input_hidden)
        clone.bias_hidden = np.copy(self.bias_hidden)
        clone.weights_hidden_output = np.copy(self.weights_hidden_output)
        clone.bias_output = np.copy(self.bias_output)
        return clone

    def mutate(self, rate=cfg.MUTATION_RATE, amount=cfg.MUTATION_AMOUNT):
        def mutate_array(arr):
            mutation_mask = np.random.rand(*arr.shape) < rate
            random_mutation = (np.random.rand(*arr.shape) - 0.5) * amount * 2
            arr += mutation_mask * random_mutation
        mutate_array(self.weights_input_hidden)
        mutate_array(self.bias_hidden)
        mutate_array(self.weights_hidden_output)
        mutate_array(self.bias_output)

    @staticmethod
    def crossover(brain1, brain2, child_brain):
        mask = np.random.rand(*brain1.weights_input_hidden.shape) > 0.5
        child_brain.weights_input_hidden = np.where(mask, brain1.weights_input_hidden, brain2.weights_input_hidden)
        mask = np.random.rand(*brain1.bias_hidden.shape) > 0.5
        child_brain.bias_hidden = np.where(mask, brain1.bias_hidden, brain2.bias_hidden)
        mask = np.random.rand(*brain1.weights_hidden_output.shape) > 0.5
        child_brain.weights_hidden_output = np.where(mask, brain1.weights_hidden_output, brain2.weights_hidden_output)
        mask = np.random.rand(*brain1.bias_output.shape) > 0.5
        child_brain.bias_output = np.where(mask, brain1.bias_output, brain2.bias_output)
from dnn.optimizers import Optimizer


class Model:
    def __init__(self, ip_layer, op_layer):
        self.ip_layer = ip_layer
        self.op_layer = op_layer

        self.layers = self._deconstruct(ip_layer, op_layer)

        self.trainable_layers = [layer for layer in self.layers if layer.trainable]

        self.opt = None

    @staticmethod
    def _deconstruct(ip_layer, op_layer):
        layers = []

        layer = op_layer

        while layer is not ip_layer:
            layers.append(layer)
            layer = layer.ip_layer

        return layers[::-1]

    def __str__(self):
        layers = ", ".join([str(l) for l in self.layers])
        return f"{self.__class__.__name__}({self.ip_layer}, {layers})"

    def __repr__(self):
        return self.__str__()

    def predict(self, X):
        self.ip_layer.ip = X

        for layer in self.layers:
            layer.forward_step()

        self.ip_layer.ip = None

        return self.op_layer.output()

    def compile(self, opt):
        if not isinstance(opt, Optimizer):
            raise ValueError("opt should be an instance of a subclass of Optimizer.")

        for layer in self.layers:
            layer.build()

        self.opt = opt

    def set_operation_mode(self, training):
        for layer in self.layers:
            layer.is_training = training

    def train(
        self, X, Y, batch_size, epochs, *args, loss="bce", shuffle=True, **kwargs
    ):
        if X.shape[-1] != Y.shape[-1]:
            raise ValueError("X and Y should have the same number of samples.")

        if Y.shape[0] != self.layers[-1].units:
            msg = "Y should have the same number of rows as number of units in the final layer."
            raise ValueError(msg)

        self.set_operation_mode(training=True)

        history = self.opt.optimize(
            self, X, Y, batch_size, epochs, *args, loss=loss, shuffle=shuffle, **kwargs
        )

        self.set_operation_mode(training=False)

        return history

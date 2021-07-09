# Deep Neural Networks From Scratch

[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/MalayAgr/DeepNeuralNetwork-Scratch)

This is an implementation of deep neural networks using nothing but Python and NumPy. I've taken up this project to complement the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), offered by Coursera and taught by Andrew Ng.

Currently, the following things are supported:

- `Dense`
- `Conv2D`
- `MaxPool` and `AveragePool`
- `BatchNorm`
- `Dropout`
- Activations: `Sigmoid`, `TanH`, `ReLU`, `LeakyReLU`, `ELU` and `Softmax`
- Losses: `BinaryCrossEntropy`, `CategoricalCrossEntropy` and `MeanSquaredError`
- Optimizers: `SGD` with no momentum, `SGD` with momentum, `RMSProp` and `Adam`
- L2 Regularization

It is also possible to easily add layers, activations, losses and optimizers.

**Note**: There is no automatic differentiation. Instead, users, when extending, need to define the necessary derivatives for backpropagation.

Hope you like it! Happy learning!

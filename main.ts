import Dataset from "./lib/dataset"
import DenseLayer from "./lib/dense_layer";
import OutputLayer from "./lib/output_layer";
import Model from "./lib/model"
import Activations from "./lib/activations";

let dataset = new Dataset();

dataset.BATCH_SIZE = 1000
dataset.loadMnistTrain("./dataset")

let layers = [
    new DenseLayer(32, 28*28, Activations.sigmoid, Activations.sigmoid_derivative),
    new DenseLayer(32, 32, Activations.sigmoid, Activations.sigmoid_derivative),
    new DenseLayer(32, 32, Activations.sigmoid, Activations.sigmoid_derivative),
    new OutputLayer(10, 32, Activations.Softmax, Activations.sigmoid_derivative)
]

let model = new Model(layers)



model.train(dataset, 30, 0.0005)

model.save("./nn.json")
dataset.BATCH_SIZE = 1
console.log(model.predict(dataset.getBatch(0)[0].data).toString())
console.log(dataset.getBatch(0)[0].label.toString())
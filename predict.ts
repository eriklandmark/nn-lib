import Dataset, {Example} from "./lib/dataset";
import Vector from "./lib/vector";
import Model from "./lib/model";
import DenseLayer from "./lib/dense_layer";
import OutputLayer from "./lib/output_layer";
import Activations from "./lib/activations";

const model = new Model([
    new DenseLayer(32, 28*28, Activations.sigmoid, Activations.sigmoid_derivative),
    new DenseLayer(64, 32, Activations.sigmoid, Activations.sigmoid_derivative),
    new DenseLayer(32, 64, Activations.sigmoid, Activations.sigmoid_derivative),
    new OutputLayer(10, 32, Activations.sigmoid, Activations.sigmoid_derivative)
])

model.load("./nn.json")

const dataset = new Dataset();
dataset.loadMnist("dataset", 1);

let ex = dataset.getBatch(0)[0]

console.log(model.predict(ex.data).toString())
console.log(ex.label.toString())
import Dataset, {Example} from "./lib/dataset"
import DenseLayer from "./lib/dense_layer";
import OutputLayer from "./lib/output_layer";
import Vector from "./lib/vector";
import Model from "./lib/model"
import Activations from "./lib/activations";

let data: Array<Example> = [
    /*{
        data: new Vector([0.05, 0.1]),
        label: new Vector([0.01, 0.99])
    },*/
    {
        data: new Vector([1, 0]),
        label: new Vector([0, 1])
    },
    {
        data: new Vector([0, 1]),
        label: new Vector([0, 1])
    },
    {
        data: new Vector([1, 1]),
        label: new Vector([1, 0])
    },
    {
        data: new Vector([0, 0]),
        label: new Vector([1, 0])
    }
]

let dataset = new Dataset();

dataset.BATCH_SIZE = 10
//dataset.loadMnist("./dataset", 1000)

let layers = [
    new DenseLayer(4, 2, Activations.sigmoid, Activations.sigmoid_derivative),
    new DenseLayer(4, 4, Activations.sigmoid, Activations.sigmoid_derivative),
    new OutputLayer(2, 4, Activations.Softmax, Activations.Softmax)
]

let model = new Model(layers)

model.train(data, 10000, 0.001)
model.save("./nn.json")

console.log(model.predict(new Vector([1,0])).toString())
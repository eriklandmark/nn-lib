import Dataset, {Example} from "./lib/dataset";
import Vector from "./lib/vector";
import Model from "./lib/model";
import DenseLayer from "./lib/dense_layer";
import OutputLayer from "./lib/output_layer";

const model = new Model([
    new DenseLayer(32, 28*28),
    new DenseLayer(32, 32),
    new DenseLayer(32, 32),
    new OutputLayer(10, 32)
])

model.load("./nn.json")

const dataset = new Dataset();
dataset.loadMnist("dataset", 1);

let ex = dataset.getBatch(0)[0]

console.log(model.predict(ex.data).toString())
console.log(ex.label.toString())
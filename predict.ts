import Dataset from "./lib/dataset";
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

model.load("./nn.old.json")

const dataset = new Dataset();
dataset.loadMnistTest("dataset/test", 10000);
dataset.BATCH_SIZE = 10000

let examples = dataset.getBatch(0)

let numRights = 0;

for (let i = 0; i < dataset.size(); i++ ) {
    const predArg = model.predict(examples[i].data).argmax(0)
    const labelArg = examples[i].label.argmax();
    if (predArg == labelArg) {
        numRights += 1
    }
}

console.log("Num rights: " + numRights + " of 10000 (" + Math.round((numRights / 10000) * 100) + " %)")

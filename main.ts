import Dataset from "./lib/dataset"
import DenseLayer from "./lib/dense_layer";
import OutputLayer from "./lib/output_layer";
import Model from "./lib/model"
import Losses from "./lib/losses";

let dataset = new Dataset();

dataset.BATCH_SIZE = 1000
dataset.loadMnistTrain("./dataset/mnist")

let layers = [
    new DenseLayer(32,"sigmoid"),
    new DenseLayer(32,"sigmoid"),
    new DenseLayer(32,"sigmoid"),
    new OutputLayer(10,"softmax")
]

let model = new Model(layers)
model.USE_GPU = false

model.build(28*28, Losses.squared_error_derivative)

run()
async function run() {
    await model.train(dataset, 30, 0.0005)
    console.log("Done")
    //model.save("./nn.json")

    /*const testDataset = new Dataset();
    testDataset.loadMnistTest("dataset/test", 10000);
    testDataset.BATCH_SIZE = 10000

    let examples = testDataset.getBatch(0)

    let numRights = 0;

    for (let i = 0; i < testDataset.size(); i++ ) {
        const predArg = model.predict(examples[i].data).argmax(0)
        const labelArg = examples[i].label.argmax();
        if (predArg == labelArg) {
            numRights += 1
        }
    }

    console.log("Num rights: " + numRights + " of 10000 (" + Math.round((numRights / 10000) * 100) + " %)")*/
}


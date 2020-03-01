import Dataset, {Example} from "./lib/dataset"
import DenseLayer from "./lib/dense_layer";
import OutputLayer from "./lib/output_layer";
import Model from "./lib/model"
import Activations from "./lib/activations";
import * as fs from "fs";
import * as path from "path";
import Vector from "./lib/vector";

let dataset = new Dataset();

dataset.BATCH_SIZE = 100
// dataset.loadMnistTrain("./dataset")
dataset.IS_GENERATOR = true

const train_images: string[] = fs.readFileSync("./dataset/speech/train_list.txt", {encoding: "UTF-8"})
    .trim().split("\n").map((s: string) => s.trim())

dataset.TOTAL_EXAMPLES = train_images.length

dataset.setGenerator(async (batch_id: number) => {
    let examples: Example[] = []

    for (let i = batch_id * dataset.BATCH_SIZE; i < batch_id*dataset.BATCH_SIZE + dataset.BATCH_SIZE; i++) {
        const file = train_images[i].replace("wav", "png")
        let label = -1;
        if (file.startsWith("_background")) {
            label = 0;
        } else if (file.startsWith("down")) {
            label = 1;
        } else if (file.startsWith("left")) {
            label = 2;
        } else if (file.startsWith("right")) {
            label = 3;
        } else if (file.startsWith("up")) {
            label = 4;
        }

        examples.push({
            data: await Dataset.read_image(path.join("./dataset/speech", file)),
            label: Vector.toCategorical(label, 5)
        })
    }

    return examples
})

let layers = [
    new DenseLayer(64, 128*118, Activations.sigmoid, Activations.sigmoid_derivative),
    new DenseLayer(32, 64, Activations.sigmoid, Activations.sigmoid_derivative),
    new DenseLayer(24, 32, Activations.sigmoid, Activations.sigmoid_derivative),
    new OutputLayer(5, 24, Activations.Softmax, Activations.sigmoid_derivative)
]

let model = new Model(layers)

run()

async function run() {
    await model.train(dataset, 30, 0.0001)

    model.save("./nn.json")

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


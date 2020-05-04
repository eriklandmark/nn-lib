import Dataset, {Example} from "../src/dataset"
import DenseLayer from "../src/layers/dense_layer";
import OutputLayer from "../src/layers/output_layer";
import Model from "../src/model"
import Softmax from "../src/activations/softmax";
import ReLu from "../src/activations/relu";
import CrossEntropy from "../src/losses/cross_entropy";
import ConvolutionLayer from "../src/layers/conv_layer";
import PoolingLayer from "../src/layers/pooling_layer";
import FlattenLayer from "../src/layers/flatten_layer";
import HyperbolicTangent from "../src/activations/hyperbolic_tangent";
import Adam from "../src/optimizers/Adam";
import * as path from "path";
import Vector from "../src/vector";
import fs from "fs"
import Tensor from "../src/tensor";

const dataset = new Dataset()

const train_images: string[] = fs.readFileSync("./dataset/speech/test_list.txt", {encoding: "UTF-8"})
    .trim().split("\n").map((s: string) => s.trim())//.slice(0, 10)

dataset.BATCH_SIZE =  30
dataset.IS_GENERATOR = true
dataset.DATA_STRUCTURE = Tensor
dataset.TOTAL_EXAMPLES = train_images.length

dataset.setGenerator(async (batch_id: number, shuffle) => {
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

const model = new Model([
    new ConvolutionLayer(8,[5,5], false, new ReLu(), true),
    new PoolingLayer([2,2], [2,2]),
    new ConvolutionLayer(12, [5,5], false, new ReLu(), true),
    new FlattenLayer(),
    new DenseLayer(512, new HyperbolicTangent()),
    new OutputLayer(5, new Softmax())
])

model.settings.USE_GPU = false
model.settings.MODEL_SAVE_PATH = "./model"
model.settings.BACKLOG = true
model.settings.SAVE_CHECKPOINTS = true
model.settings.VERBOSE_COMPACT = true
model.settings.EVAL_PER_EPOCH = false

model.build([118,128,1], 0.001, CrossEntropy, Adam)
model.summary()

async function run() {
    await model.train(dataset, 30, null, true)
    model.save()
    let ex = dataset.getBatch(0)[0]
    console.log(model.predict(ex.data).toString())
    console.log(ex.label.toString())
}

run()
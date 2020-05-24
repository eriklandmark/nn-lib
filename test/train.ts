import DenseLayer from "../src/layers/dense_layer";
import OutputLayer from "../src/layers/output_layer";
import Model from "../src/model"
import Softmax from "../src/activations/softmax";
import CrossEntropy from "../src/losses/cross_entropy";
import Sigmoid from "../src/activations/sigmoid";
import Dataset from "../src/dataset";
import Adam from "../src/optimizers/Adam";
import ReLu from "../src/activations/relu";
import ConvolutionLayer from "../src/layers/conv_layer";
import PoolingLayer from "../src/layers/pooling_layer";
import FlattenLayer from "../src/layers/flatten_layer";

console.log("Starting..")
const dataset = new Dataset()

dataset.loadMnistTrain("./dataset/mnist", 1000, false)

dataset.BATCH_SIZE = 50
dataset.TOTAL_EXAMPLES = dataset.size()
dataset.IS_GENERATOR = false
dataset.DATA_SHAPE = [28,28,1]

const eval_dataset = new Dataset()
eval_dataset.loadMnistTest("./dataset/mnist", 200, false)

eval_dataset.BATCH_SIZE = 200
eval_dataset.TOTAL_EXAMPLES = dataset.size()
eval_dataset.IS_GENERATOR = false
eval_dataset.DATA_SHAPE = [28, 28, 1]
eval_dataset.VERBOSE = false

const model = new Model([
    new ConvolutionLayer(8, [5,5], false, new ReLu()),
    new ConvolutionLayer(12, [5,5], false, new ReLu()),
    new PoolingLayer([2,2], [2,2]),
    new ConvolutionLayer(16, [5,5], false, new ReLu()),
    new ConvolutionLayer(24, [5,5], false, new ReLu()),
    new FlattenLayer(),
    new DenseLayer(64, new Sigmoid()),
    new OutputLayer(10, new Softmax())
])

model.settings.USE_GPU = false
model.settings.MODEL_SAVE_PATH = "./model_nlp"
model.settings.BACKLOG = true
model.settings.EVAL_PER_EPOCH = true
model.settings.SAVE_CHECKPOINTS = true

model.build(dataset.DATA_SHAPE,0.01, CrossEntropy, Adam)
model.summary()

async function run() {
    await model.train(dataset, 2, eval_dataset, false)
    console.log("Done")
    model.save()
    let ex = dataset.getBatch(0)[0]
    console.log(model.predict(ex.data).toString())
    console.log(ex.label.toString())
}

//run()
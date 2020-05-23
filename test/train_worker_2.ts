import Dataset from "../src/dataset";
import Model from "../src/model";
import ConvolutionLayer from "../src/layers/conv_layer";
import ReLu from "../src/activations/relu";
import PoolingLayer from "../src/layers/pooling_layer";
import FlattenLayer from "../src/layers/flatten_layer";
import DenseLayer from "../src/layers/dense_layer";
import Sigmoid from "../src/activations/sigmoid";
import OutputLayer from "../src/layers/output_layer";
import Softmax from "../src/activations/softmax";
import CrossEntropy from "../src/losses/cross_entropy";
import Adam from "../src/optimizers/Adam";

const dataset = new Dataset()

dataset.loadMnistTrain("./dataset/mnist", 1000, false)

dataset.BATCH_SIZE = 50
dataset.TOTAL_EXAMPLES = dataset.size()
dataset.IS_GENERATOR = false
dataset.DATA_SHAPE = [28, 28, 1]
dataset.VERBOSE = false

const EPOCHS = 100

const model = new Model([
    new ConvolutionLayer(8, [5, 5], false, new ReLu()),
    new PoolingLayer([2, 2], [2, 2]),
    new ConvolutionLayer(16, [5, 5], false, new ReLu()),
    new FlattenLayer(),
    new DenseLayer(700, new Sigmoid()),
    new OutputLayer(10, new Softmax())
])

process.on("message", (data) => {
    if (data["action"] == "run") {
        model.train(dataset, EPOCHS, null, true)
    } else if (data["action"] == "build") {
        model.settings.USE_GPU = false
        model.settings.MODEL_SAVE_PATH = "./model_" + data["id"]
        model.settings.BACKLOG = true
        model.settings.EVAL_PER_EPOCH = false
        model.settings.SAVE_CHECKPOINTS = true
        model.settings.WORKER_MODE = true
        model.settings.WORKER_CALLBACK = (cData) => {
            cData["id"] = data["id"];
            process.send({action: "update", data: cData})
        }

        model.build(dataset.DATA_SHAPE, 0.01, CrossEntropy, Adam, false)

        process.send({action: "built", total_steps: ((dataset.TOTAL_EXAMPLES / dataset.BATCH_SIZE)* EPOCHS)})
    }
})
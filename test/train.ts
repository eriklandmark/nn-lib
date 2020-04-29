import Dataset from "../src/dataset"
import DenseLayer from "../src/layers/dense_layer";
import OutputLayer from "../src/layers/output_layer";
import Model from "../src/model"
import Softmax from "../src/activations/softmax";
import ReLu from "../src/activations/relu";
import CrossEntropy from "../src/losses/cross_entropy";
import Vector from "../src/vector";
import StochasticGradientDescent from "../src/optimizers/StochasticGradientDescent";
import ConvolutionLayer from "../src/layers/conv_layer";
import PoolingLayer from "../src/layers/pooling_layer";
import FlattenLayer from "../src/layers/flatten_layer";
import HyperbolicTangent from "../src/activations/hyperbolic_tangent";
import Adam from "../src/optimizers/Adam";

const dataset = new Dataset()

dataset.BATCH_SIZE = 50
dataset.loadMnistTrain("./dataset/mnist", 1000, false)

const model = new Model([
    new ConvolutionLayer(8,[5,5], false, new ReLu()),
    new PoolingLayer([2,2], [2,2]),
    new ConvolutionLayer(16, [5,5], false, new ReLu()),
    new FlattenLayer(),
    new DenseLayer(512, new HyperbolicTangent()),
    new OutputLayer(10, new Softmax())
])

model.settings.USE_GPU = false
model.settings.MODEL_SAVE_PATH = "./model"
model.settings.BACKLOG = true
model.settings.SAVE_CHECKPOINTS = true
model.settings.VERBOSE_COMPACT = true

model.build([28,28,1], 0.001, CrossEntropy, Adam)
model.summary()

async function run() {
    await model.train(dataset, 100, true)
    model.save()
    let ex = dataset.getBatch(0)[0]
    console.log(model.predict(ex.data).toString())
    console.log(ex.label.toString())
}

run()
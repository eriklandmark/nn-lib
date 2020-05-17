import Dataset, {Example} from "../src/dataset"
import DenseLayer from "../src/layers/dense_layer";
import OutputLayer from "../src/layers/output_layer";
import Model from "../src/model"
import Softmax from "../src/activations/softmax";
import CrossEntropy from "../src/losses/cross_entropy";
import ConvolutionLayer from "../src/layers/conv_layer";
import PoolingLayer from "../src/layers/pooling_layer";
import FlattenLayer from "../src/layers/flatten_layer";
import HyperbolicTangent from "../src/activations/hyperbolic_tangent";
import Adam from "../src/optimizers/Adam";
import ReLu from "../src/activations/relu";
import DropoutLayer from "../src/layers/dropout_layer";
import Tokenizer from "../src/linguistics/tokenizer";
import CsvParser from "../src/linguistics/csv_parser";
import ArrayHelper from "../src/lib/array_helper";
import Sigmoid from "../src/activations/sigmoid";
import Vector from "../src/vector";
import StochasticGradientDescent from "../src/optimizers/StochasticGradientDescent"

console.log("Starting..")
const tokenizer = new Tokenizer()

const trainData = CsvParser.parse("./dataset/nlp/train.tsv", true)
const sentences = ArrayHelper.flatten(CsvParser.filterColumns(trainData, [3]))
tokenizer.createVocabulary(sentences)

const dataset = tokenizer.createDataset("./dataset/nlp/train.tsv", [1,3])

dataset.BATCH_SIZE = dataset.size()
dataset.TOTAL_EXAMPLES = dataset.size()
dataset.IS_GENERATOR = false
dataset.DATA_STRUCTURE = Vector

const model = new Model([
    new DenseLayer(8, new Sigmoid()),
    new DenseLayer(16, new Sigmoid()),
    new DenseLayer(16, new Sigmoid()),
    new OutputLayer(3, new Softmax())
])

model.settings.USE_GPU = false
model.settings.MODEL_SAVE_PATH = "./model_nlp"
model.settings.BACKLOG = true
model.settings.EVAL_PER_EPOCH = false
model.settings.SAVE_CHECKPOINTS = false

model.build([4],0.01, CrossEntropy, StochasticGradientDescent)
model.summary()

async function run() {
    await model.train(dataset, 300, null, false)
    console.log("Done")
    model.save()
    let ex = dataset.getBatch(0)[0]
    console.log(model.predict(ex.data).toString())
    console.log(ex.label.toString())
}

run()
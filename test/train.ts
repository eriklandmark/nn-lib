import Dataset from "../src/dataset"
import DenseLayer from "../src/lib/layers/dense_layer";
import OutputLayer from "../src/lib/layers/output_layer";
import Model from "../src/model"
import Sigmoid from "../src/lib/activations/sigmoid";
import Softmax from "../src/lib/activations/softmax";
import ConvolutionLayer from "../src/lib/layers/conv_layer";
import FlattenLayer from "../src/lib/layers/flatten_layer";
import DropoutLayer from "../src/lib/layers/dropout_layer";
import ReLu from "../src/lib/activations/relu";
import CrossEntropy from "../src/lib/losses/cross_entropy";

let dataset = new Dataset();

dataset.BATCH_SIZE = 25
dataset.loadMnistTrain("./dataset/mnist-fashion", 100, false)
let layers = [
    new ConvolutionLayer(8, [5,5], false, new ReLu()),
    new ConvolutionLayer(16, [5,5], false, new ReLu()),
    new FlattenLayer(),
    new DenseLayer(500, new Sigmoid()),
    new DenseLayer(100, new Sigmoid()),
    //new DropoutLayer(0.1),
    new OutputLayer(10, new Softmax())
]

let model = new Model(layers)
model.USE_GPU = false

model.build([28,28,3], new CrossEntropy())
model.summary()

async function run() {
    await model.train(dataset, 40, 0.0001)
    console.log("Done")
    model.save("./nn.json")
    let ex = dataset.getBatch(0)[0]
    console.log(model.predict(ex.data).toString())
    console.log(ex.label.toString())
}

run()
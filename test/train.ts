import Dataset from "../src/dataset"
import DenseLayer from "../src/lib/layers/dense_layer";
import OutputLayer from "../src/lib/layers/output_layer";
import Model from "../src/model"
import Sigmoid from "../src/lib/activations/sigmoid";
import Softmax from "../src/lib/activations/softmax";
import MeanSquaredError from "../src/lib/losses/mean_squared_error";
import ConvolutionLayer from "../src/lib/layers/conv_layer";
import FlattenLayer from "../src/lib/layers/flatten_layer";
import Tensor from "../src/tensor";

let dataset = new Dataset();

dataset.BATCH_SIZE = 10
dataset.loadMnistTrain("./dataset/mnist", 100, false)

let layers = [
    new ConvolutionLayer(4, [3,3], new Sigmoid()),
    new FlattenLayer(),
    new DenseLayer(32, new Sigmoid()),
    new OutputLayer(10, new Softmax())
]

let model = new Model(layers)
model.USE_GPU = false

model.build([28, 28, 3], new MeanSquaredError())

async function run() {
    await model.train(dataset.getBatch(0), 500, 0.0005)
    console.log("Done")
    model.save("./nn.json")
    let ex = dataset.getBatch(0)[0]
    console.log(model.predict([<Tensor> ex.data]).toString())
    console.log(ex.label.toString())
}
run()
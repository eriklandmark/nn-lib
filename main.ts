import Dataset, {Example} from "./lib/dataset"
import DenseLayer from "./lib/dense_layer";
import OutputLayer from "./lib/output_layer";
import Vector from "./lib/vector";
import Model from "./lib/model"
import Activations from "./lib/activations";

let dataset = new Dataset();

dataset.BATCH_SIZE = 2100
dataset.loadTestData("./data.json")

let layers = [
    new DenseLayer(4, 2, Activations.sigmoid, Activations.sigmoid_derivative),
    new OutputLayer(3, 4, Activations.Softmax, Activations.sigmoid_derivative)
]

let model = new Model(layers)

model.train(dataset, 1000, 0.01)

//model.save("./nn.json")

//console.log(model.predict(new Vector([1,0])).toString())
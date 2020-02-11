import Dataset, {Example} from "./lib/dataset"
import DenseLayer from "./lib/DenseLayer";
import Matrix from "./lib/matrix";
import Vector from "./lib/vector";
import Activations from "./lib/activations";
import vector from "./lib/vector";
import Losses from "./lib/losses";


let dataset = new Dataset();

const MAX_EXAMPLES = 1000;
const BATCH_SIZE = 10;
const EPOCHS = 10
const BATCHES_PER_EPOCH = MAX_EXAMPLES / BATCH_SIZE;
const LEARNING_RATE = 0.1;

dataset.loadMnist("./dataset", MAX_EXAMPLES)

//let inputLayer = new DenseLayer(784, 32)
//let hiddenLayer1 = new DenseLayer(32, 32)
//let hiddenLayer2 = new DenseLayer(32, 32)
//let outputLayer = new DenseLayer(32, 10)

let w1 = new Matrix();
w1.createEmptyArray(784, 32)
w1.populateRandom();

let w2 = new Matrix();
w2.createEmptyArray(32, 32)
w2.populateRandom();

let w3 = new Matrix();
w3.createEmptyArray(32, 10)
w3.populateRandom();

let b1 = new Vector(32);
b1.populateRandom();

let b2 = new Vector(32);
b2.populateRandom();

let b3 = new Vector(10);
b3.populateRandom();


for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for(let batchNr = 0; batchNr < BATCHES_PER_EPOCH; batchNr++) {
        let batch: Array<Example> = dataset.getBatch(batchNr, BATCH_SIZE)
        for(let example of batch) {
            const z1 = (<Vector> w1.mm(example.data)).add(b1)
            const a1 = Activations.sigmoid(z1)
            const z2 = (<Vector> w2.mm(a1)).add(b2)
            const a2 = Activations.sigmoid(z2)
            const z3 = (<Vector> w3.mm(a2)).add(b3)
            const a3 = Activations.sigmoid(z3)

            const error = Losses.defLoss(a3, example.label)
            const gradient: number = Losses.defLoss_derivative(a3, example.label)



        }
    }
}

/*
for (let epoch = 0; epoch < EPOCHS; epoch++) {
    for(let batchNr = 0; batchNr < BATCHES_PER_EPOCH; batchNr++) {
        let batch: Array<Example> = dataset.getBatch(batchNr, BATCH_SIZE)
        for(let example of batch) {
            const z1 = (<Vector> w1.mm(example.data)).add(b1)
            const a1 = Activations.sigmoid(z1)
            const z2 = (<Vector> w2.mm(a1)).add(b2)
            const a2 = Activations.sigmoid(z2)
            const z3 = (<Vector> w3.mm(a2)).add(b3)
            const a3 = Activations.sigmoid(z3)

            const error = Losses.defLoss(a3, example.label)
            const gradient: number = Losses.defLoss_derivative(a3, example.label)

            const e3 = Activations.sigmoid_derivative(z3).mul(gradient)
            const e2 = (<Vector> w3.transpose().mm(e3)).mul(Activations.sigmoid_derivative(z2))
            const e1 = (<Vector> w2.transpose().mm(e2)).mul(Activations.sigmoid_derivative(z1))

            b3 = b3.sub(e3.mean() * LEARNING_RATE)
            b2 = b2.sub(e2.mean() * LEARNING_RATE)
            b1 = b1.sub(e1.mean() * LEARNING_RATE)

        }
    }
}
*/
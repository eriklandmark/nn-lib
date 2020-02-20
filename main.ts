import Dataset, {Example} from "./lib/dataset"
import DenseLayer from "./lib/DenseLayer";
import Matrix from "./lib/matrix";
import Vector from "./lib/vector";
import Activations from "./lib/activations";
import Losses from "./lib/losses";
import * as fs from "fs";


let dataset = new Dataset();

const MAX_EXAMPLES = 1000;
const BATCH_SIZE = 10;
const EPOCHS = 10
const BATCHES_PER_EPOCH = MAX_EXAMPLES / BATCH_SIZE;
const LEARNING_RATE = 0.1;

//dataset.loadMnist("./dataset", MAX_EXAMPLES)

//let inputLayer = new DenseLayer(784, 32)
//let hiddenLayer1 = new DenseLayer(32, 32)
//let hiddenLayer2 = new DenseLayer(32, 32)
//let outputLayer = new DenseLayer(32, 10)
/*
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
*/

// Weight matrix är nästa lagers antal noder X nuvarandes lagers nod

//let h_w = new Matrix([[0.15, 0.2], [0.25, 0.30]]);
let h_w = new Matrix();
h_w.createEmptyArray(8, 2)
h_w.populateRandom();

//let h_b = new Vector([0.35, 0.35]);
let h_b = new Vector(8);
h_b.populateRandom();

//let o_w = new Matrix([[0.4, 0.45], [0.50, 0.55]]);
let o_w = new Matrix();
o_w.createEmptyArray(2, 8)
o_w.populateRandom();

//let o_b = new Vector([0.6,  0.6]);
let o_b = new Vector(2);
o_b.populateRandom();

//console.log("w", h_w.matrix)
//console.log("w", o_w.matrix)
//console.log("w", h_b.toString())
//console.log("w", o_b.toString())



let data: Array<Example> = [
    /*{
        data: new Vector([0.05, 0.1]),
        label: new Vector([0.01, 0.99])
    },*/
    {
        data: new Vector([1, 0]),
        label: new Vector([1, 0])
    },
    {
        data: new Vector([0, 1]),
        label: new Vector([0, 1])
    }/*,
    {
        data: new Vector([1, 1]),
        label: new Vector([0, 1])
    },
    {
        data: new Vector([0, 0]),
        label: new Vector([0, 1])
    }*/
]

function save() {
    const saveObject = {
        "layer_1": {
            "weights": h_w.matrix,
            "biases": h_b.vector
        }, "output_layer": {
            "weights": o_w.matrix,
            "biases": o_b.vector
        }
    }

    const model = JSON.stringify(saveObject)
    fs.writeFileSync("./nn.json", model)
}

function train(example: Example) {
    //console.log("Labels", example.label.toString())
    // Feed forward
    //console.log(o_w.toString())
    const z1 = (<Vector> h_w.mm(example.data)).add(h_b)
    const a1 = <Vector> Activations.sigmoid(z1)
    //console.log("Activation 1", a1.toString())
    const z2 = (<Vector> o_w.mm(a1)).add(o_b)
    const a2 = Activations.sigmoid(z2)
    //console.log("Activation 2", a2.toString())

    //Backpropagation

    const errorVector = <Vector> Losses.squared_error(a2, example.label)
    const errorSum = errorVector.sum()
    //console.log("Error", errorVector.toString())
    console.log("Error sum:", errorSum)

    const dError = <Vector> Losses.squared_error_derivative(a2, example.label)
    //console.log("dError/dOut", dError.toString())

    const dA2dz2 = <Vector> Activations.sigmoid_derivative(a2)
    //console.log("dA2/z2", dA2dz2.toString())
    const dZ2dWh = <Vector> a2;

    const deltaErrorW2 = dError.mul(dA2dz2).mul(dZ2dWh)
    //console.log(deltaErrorW2.toString())

    const deltaW2 = o_w.copy()
    deltaW2.iterate((i, j) => {
        deltaW2.set(i, j, deltaW2.get(i,j) - (LEARNING_RATE * deltaErrorW2.get(i)));
    })
    //console.log(deltaW2.toString())

    // Hidden Layer

    const dEo1dOuth1 = dError.mul(dA2dz2)
    //console.log("dEo1dOuth1", dEo1dOuth1.toString())

    const dEtotdOuth = <Vector> o_w.transpose().mm(dEo1dOuth1)
    //console.log(dEtotdOuth.toString())

    const dOuthdNeth = <Vector> Activations.sigmoid_derivative(a1)
    //console.log(dOuthdNeth.toString())

    const deltaErrorW1 = dEtotdOuth.mul(dOuthdNeth).mul(a1)
    //console.log(deltaErrorW1.toString())

    const deltaW1 = h_w.copy()
    deltaW1.iterate((i: number, j: number) => {
        deltaW1.set(i, j, deltaW1.get(i,j) - (LEARNING_RATE * deltaErrorW1.get(i)));
    })

    //console.log(deltaW1.toString())

    o_w = deltaW2
    h_w = deltaW1


    /*
    const errorHiddenVector = <Vector>o_w.transpose().mm(errorVector)
    //console.log("Error", errorVector.toString())
    //console.log("Hidden Error", errorHiddenVector.toString())
    const gradient_w_o: Vector = Activations.sigmoid_derivative(a2).mul(errorVector).mul(LEARNING_RATE)
    const dw_o: Matrix = <Matrix> new Matrix([gradient_w_o]).mm(new Matrix([a1]).transpose())
    //console.log("Gradient Output", gradient_w_o.toString())
    //console.log("delta Weight for Output", dw_o.toString())

    const gradient_w_h: Vector = Activations.sigmoid_derivative(a1).mul(errorHiddenVector).mul(LEARNING_RATE)
    const dw_h: Matrix = <Matrix> new Matrix([gradient_w_h]).mm(new Matrix([example.data]).transpose())
    console.log("delta Weight for Hidden", dw_h.toString())

    o_w.add(dw_o)
    h_w.add(dw_h)
    o_b.add(gradient_w_o)
    h_b.add(gradient_w_h)*/
}


for (let epoch = 0; epoch < 10000; epoch++) {
    //train(data[0])
    data = shuffle(data)
    for (let example of data) {
        train(example)
    }
}

function predict(example: Example) {
    const z1 = (<Vector> h_w.mm(example.data)).add(h_b)
    const a1 = Activations.sigmoid(z1)
    const z2 = (<Vector> o_w.mm(a1)).add(o_b)
    const a2 = Activations.sigmoid(z2)
    return a2
}

//console.log(data[0].label.toString())
//console.log("Predicted: ", predict(data[0]).toString())


//save()
console.log("Done")


function shuffle(array) {
    let currentIndex = array.length, temporaryValue, randomIndex;

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {

        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }

    return array;
}
/*
function train(example: Example) {
    //console.log("Labels", example.label.toString())
    // Feed forward
    const z1 = (<Vector> h_w.mm(example.data)).add(h_b)
    const a1 = Activations.sigmoid(z1)
    const z2 = (<Vector> o_w.mm(a1)).add(o_b)
    const a2 = Activations.sigmoid(z2)
    //console.log("Activation", a2.toString())

    //Backpropagation
    const errorVector = Losses.defLoss(a2, example.label)
    //console.log("Error", errorVector.toString())

    const errorHiddenVector: Vector = <Vector>o_w.transpose().mm(errorVector)
    //console.log("Error", errorVector.toString())
    //console.log("Hidden Error", errorHiddenVector.toString())
    const gradient_w_o: Vector = Activations.sigmoid_derivative(a2).mul(errorVector).mul(LEARNING_RATE)
    const dw_o: Matrix = <Matrix> new Matrix([gradient_w_o]).mm(new Matrix([a1]).transpose())
    //console.log("Gradient Output", gradient_w_o.toString())
    //console.log("delta Weight for Output", dw_o.toString())

    const gradient_w_h: Vector = Activations.sigmoid_derivative(a1).mul(errorHiddenVector).mul(LEARNING_RATE)
    const dw_h: Matrix = <Matrix> new Matrix([gradient_w_h]).mm(new Matrix([example.data]).transpose())
    console.log("delta Weight for Hidden", dw_h.toString())

    o_w.add(dw_o)
    h_w.add(dw_h)
    o_b.add(gradient_w_o)
    h_b.add(gradient_w_h)
}

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
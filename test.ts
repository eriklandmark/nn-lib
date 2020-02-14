import Matrix from "./lib/matrix";
import Vector from "./lib/vector";
import Activations from "./lib/activations";
import * as fs from "fs";
import {Example} from "./lib/dataset";

let h_w = new Matrix()
let h_b = new Vector()
let o_w = new Matrix()
let o_b = new Vector()

let data: Array<Example> = [
    {
        data: new Vector([1, 0]),
        label: new Vector([1, 0])
    },
    {
        data: new Vector([0, 1]),
        label: new Vector([1, 0])
    },
    {
        data: new Vector([1, 1]),
        label: new Vector([0, 1])
    },
    {
        data: new Vector([0, 0]),
        label: new Vector([0, 1])
    }
]

function loadModel() {
    const modelData = fs.readFileSync("./nn.json", {encoding: "UTF-8"})
    const model = JSON.parse(modelData)
    h_w = Matrix.fromJsonObject(model["layer_1"].weights)
    o_w = Matrix.fromJsonObject(model["output_layer"].weights)
    //h_b = Vector.fromJsonObject(model["layer_1"].biases)
    //o_b = Vector.fromJsonObject(model["output_layer"].biases)
}

//loadModel()

function predict(example: Example) {
    const z1 = (<Vector> h_w.mm(example.data)).add(h_b)
    const a1 = Activations.sigmoid(z1)
    const z2 = (<Vector> o_w.mm(a1)).add(o_b)
    const a2 = Activations.sigmoid(z2)
    return a2
}

//console.log(predict(data[0]).toString())

console.log(new Matrix([new Vector([1,2])]).transpose().toString())
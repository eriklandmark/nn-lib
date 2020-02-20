import {Example} from "./lib/dataset";
import Vector from "./lib/vector";
import Activations from "./lib/activations";
import Matrix from "./lib/matrix";
import * as fs from "fs";

let h_w = new Matrix()
let h_b = new Vector()
let o_w = new Matrix()
let o_b = new Vector()

function loadModel() {
    const modelData = JSON.parse(fs.readFileSync("./nn.json", {encoding: "UTF-8"}))
    h_w = new Matrix(modelData["layer_1"].weights.map((row: any) => {
        return Object.keys(row).map((item) => row[item])
    }))
    h_b = new Vector(Object.keys(modelData["layer_1"].biases).map((item) => modelData["layer_1"].biases[item]))
    o_w = new Matrix(modelData["output_layer"].weights.map((row: any) => {
        return Object.keys(row).map((item) => row[item])
    }))
    o_b = new Vector(Object.keys(modelData["output_layer"].biases).map((item) => modelData["output_layer"].biases[item]))
}

loadModel()

function predict(example: Example): Vector {
    const z1 = (<Vector> h_w.mm(example.data)).add(h_b)
    const a1 = Activations.sigmoid(z1)
    const z2 = (<Vector> o_w.mm(a1)).add(o_b)
    const a2 = <Vector> Activations.sigmoid(z2)
    return a2
}

console.log(predict({
    data: new Vector([0, 0]),
    label: new Vector([1, 0])
}).toString())



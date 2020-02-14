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
    const modelData = fs.readFileSync("./nn.json")
    console.log(modelData)
}

loadModel()

function predict(example: Example) {
    const z1 = (<Vector> h_w.mm(example.data)).add(h_b)
    const a1 = Activations.sigmoid(z1)
    const z2 = (<Vector> o_w.mm(a1)).add(o_b)
    const a2 = Activations.sigmoid(z2)
    return a2
}




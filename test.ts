import Matrix from "./lib/matrix";
import Vector from "./lib/vector";
import Activations from "./lib/activations";
import * as fs from "fs";
import {Example} from "./lib/dataset";
import MatrixHelper from "./lib/matrix_helper";
//import Plot from "./plot/plot";

let a = new Matrix([[1,3,3], [1,2,3]])
let b = new Matrix([[1,2,3]])

console.log(a.sum(1, false).toString())
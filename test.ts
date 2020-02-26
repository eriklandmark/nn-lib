import Matrix from "./lib/matrix";
import Vector from "./lib/vector";
import Activations from "./lib/activations";
import * as fs from "fs";
import Dataset, {Example} from "./lib/dataset";
import MatrixHelper from "./lib/matrix_helper";
//import Plot from "./plot/plot";

let dataset = new Dataset();

dataset.BATCH_SIZE = 1000
//dataset.loadTestData("./data.json")
dataset.loadMnist("./dataset", 1000)

const startTime = Date.now();
const dur = Date.now() - startTime
console.log(startTime / 1000 * 60 *)
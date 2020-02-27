import Matrix from "./lib/matrix";
import Dataset from "./lib/dataset";

let dataset = new Dataset();

dataset.BATCH_SIZE = 1000
//dataset.loadTestData("./data.json")
//dataset.loadMnistTest("./dataset/test", 1000)

const size = 900

const a = new Matrix()
a.createEmptyArray(size,size);
a.populateRandom()
const b = new Matrix()
b.createEmptyArray(size,size);
b.populateRandom();





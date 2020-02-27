import Matrix from "./lib/matrix";
import Dataset from "./lib/dataset";

let dataset = new Dataset();

dataset.BATCH_SIZE = 1000
//dataset.loadTestData("./data.json")
//dataset.loadMnistTest("./dataset/test", 1000)

const size = 200

const a = new Matrix()
a.createEmptyArray(size,size);
a.populateRandom()
const b = new Matrix()
b.createEmptyArray(size,size);
b.populateRandom();

async function test() {
    console.log("Starting")
    const startTime = Date.now()

    await a.mm(b)

    const dur = (Date.now() - startTime) / 1000
    console.log("Done! " + dur + " seconds")
}

test()



import Model from "../src/model";
import Dataset from "../src/dataset";
import Helper from "../src/helpers/helper";

const model = new Model([])

model.load("./nn.json")

const dataset = new Dataset();

dataset.loadMnistTest("dataset/mnist-fashion", 10000, true);
dataset.BATCH_SIZE = 10000

let examples = dataset.getBatch(0)

//const pred = model.predict(examples.data)
//console.log(pred.toString())
let numRights = 0;


Helper.timeit(() => {
    for (let i = 0; i < examples.length; i++ ) {
        const pred = model.predict(examples[i].data)
        //console.log(pred.toString())
        const predArg = pred.argmax(0)
        const labelArg = examples[i].label.argmax();
        if (predArg == labelArg) {
            numRights += 1
        }
    }
}, false).then((seconds) => {
    console.log("Num rights: " + numRights + " of " + examples.length + " (" + Math.round((numRights / examples.length) * 100) + " %)")
    console.log("It took " + seconds + " seconds.")
})




/*
const model = new Model([
    new DenseLayer(64,"sigmoid"),
    new DenseLayer(32,"sigmoid"),
    new DenseLayer(24,"sigmoid"),
    new OutputLayer(5,"softmax")
])

model.build(128*118, Losses.squared_error_derivative)

const train_images: string[] = fs.readFileSync("./dataset/speech/test_list.txt", {encoding: "UTF-8"})
    .trim().split("\n").map((s: string) => s.trim())//.slice(0, 10)

dataset.BATCH_SIZE =  train_images.length
dataset.IS_GENERATOR = true

dataset.TOTAL_EXAMPLES = train_images.length

dataset.setGenerator(async (batch_id: number) => {
    let examples: Example[] = []

    for (let i = batch_id * dataset.BATCH_SIZE; i < batch_id*dataset.BATCH_SIZE + dataset.BATCH_SIZE; i++) {
        const file = train_images[i].replace("wav", "png")

        let label = -1;
        if (file.startsWith("_background")) {
            label = 0;
        } else if (file.startsWith("down")) {
            label = 1;
        } else if (file.startsWith("left")) {
            label = 2;
        } else if (file.startsWith("right")) {
            label = 3;
        } else if (file.startsWith("up")) {
            label = 4;
        }

        examples.push({
            data: await Dataset.read_image(path.join("./dataset/speech", file)),
            label: Vector.toCategorical(label, 5)
        })
    }

    return examples
})

async function run() {
    console.log("Creating examples..")
    const examples = await dataset.GENERATOR(0)
    console.log("Done!")

    let numRights = 0;

    for (let i = 0; i < examples.length; i++ ) {
        const pred = model.predict(examples[i].data)
        //console.log(pred.toString())
        const predArg = pred.argmax(0)
        const labelArg = examples[i].label.argmax();
        if (predArg == labelArg) {
            numRights += 1
        }
    }

    console.log("Num rights: " + numRights + " of " + examples.length + " (" + Math.round((numRights / examples.length) * 100) + " %)")
}

run()

 */



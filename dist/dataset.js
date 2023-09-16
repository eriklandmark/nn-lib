var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import * as fs from "fs";
import * as path from 'path';
import Jimp from 'jimp';
import Tensor from "./tensor";
import ArrayHelper from "./lib/array_helper";
import cliProgress from "cli-progress";
export default class Dataset {
    constructor() {
        this.data = [];
        this.VERBOSE = false;
        this.BATCH_SIZE = 1;
        this.IS_GENERATOR = false;
        this.TOTAL_EXAMPLES = 0;
        this.DATA_SHAPE = [];
        this.GENERATOR = () => {
        };
    }
    size() {
        return this.data.length;
    }
    setGenerator(gen) {
        this.GENERATOR = gen;
    }
    addExample(ex) {
        this.data.push(ex);
    }
    static read_image(path, channels = 3) {
        return __awaiter(this, void 0, void 0, function* () {
            const image = yield Jimp.read(path);
            const t = new Tensor([image.getHeight(), image.getWidth(), channels], true);
            if (channels > 4) {
                channels = 4;
            }
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                for (let i = 0; i < channels; i++) {
                    t.t[y][x][i] = image.bitmap.data[idx + i];
                }
            });
            return t;
        });
    }
    loadMnistTrain(folderPath, maxExamples = 60000, vectorize = true) {
        this.loadMnist(folderPath, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", maxExamples, vectorize);
    }
    loadMnistTest(folderPath, maxExamples = 60000, vectorize = true) {
        this.loadMnist(folderPath, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", maxExamples, vectorize);
    }
    shuffle() {
        this.data = ArrayHelper.shuffle(this.data);
    }
    loadMnist(folderPath, imageFileName, labelFileName, maxExamples, vectorize) {
        const trainFileBuffer = fs.readFileSync(path.join(folderPath + "/" + imageFileName));
        const labelFileBuffer = fs.readFileSync(path.join(folderPath + "/" + labelFileName));
        this.TOTAL_EXAMPLES = maxExamples;
        let bar;
        if (this.VERBOSE) {
            bar = new cliProgress.Bar({
                barCompleteChar: '#',
                barIncompleteChar: '-',
                format: 'Loading mnist.. [' + '{bar}' + '] {percentage}% | {value}/{total}',
                fps: 10,
                stream: process.stdout,
                barsize: 30
            });
            bar.start(maxExamples, 0);
        }
        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            const size = 28;
            const image = new Tensor([size, size, 1], true);
            for (let x = 0; x < size; x++) {
                for (let y = 0; y < size; y++) {
                    image.t[y][x][0] = trainFileBuffer[(imageIndex * size * size) + (x + (y * size)) + 15];
                }
            }
            let exampleData = vectorize ? image.vectorize().div(255) : image.div(255);
            this.DATA_SHAPE = exampleData.shape;
            let example = {
                data: exampleData,
                label: Tensor.toCategorical(labelFileBuffer[imageIndex + 8], 10)
            };
            this.data.push(example);
            if (this.VERBOSE) {
                bar.increment();
            }
        }
        if (this.VERBOSE) {
            bar.stop();
        }
    }
    loadTestData(path, maxExamples = 2100) {
        const data = JSON.parse(fs.readFileSync(path, { encoding: "utf-8" }));
        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            let example = {
                data: new Tensor(data["features"][imageIndex]),
                label: Tensor.toCategorical(data["labels"][imageIndex], 3)
            };
            this.data.push(example);
        }
    }
    getBatch(batch_id) {
        if (this.IS_GENERATOR) {
            return this.GENERATOR(batch_id);
        }
        else {
            return this.data.slice(batch_id * this.BATCH_SIZE, batch_id * this.BATCH_SIZE + this.BATCH_SIZE);
        }
    }
}

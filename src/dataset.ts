import * as fs from "fs";
import * as path from 'path';
import Jimp from 'jimp';
import Tensor from "./tensor";
import ArrayHelper from "./lib/array_helper";
import cliProgress from "cli-progress"

export interface Example {
    data: Tensor,
    label: Tensor
}

export default class Dataset {
    private data: Array<Example> = []

    public VERBOSE = false
    public BATCH_SIZE = 1;
    public IS_GENERATOR = false;
    public TOTAL_EXAMPLES = 0;
    public DATA_SHAPE: number[] = []
    public GENERATOR: Function = () => {
    };

    public size(): number {
        return this.data.length;
    }

    public setGenerator(gen: Function) {
        this.GENERATOR = gen
    }

    public addExample(ex: Example) {
        this.data.push(ex)
    }

    public static async read_image(path: string, channels:number = 3): Promise<Tensor> {
        const image = await Jimp.read(path);
        const t = new Tensor([image.getHeight(), image.getWidth(), channels], true)
        if (channels > 4) {
            channels = 4
        }
        image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
            for( let i = 0; i < channels; i++) {
                t.t[y][x][i] = image.bitmap.data[idx + i]
            }
        });
        return t
    }

    public loadMnistTrain(folderPath: string, maxExamples: number = 60000, vectorize: boolean = true) {
        this.loadMnist(folderPath, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", maxExamples, vectorize)
    }

    public loadMnistTest(folderPath: string, maxExamples: number = 60000, vectorize: boolean = true) {
        this.loadMnist(folderPath, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", maxExamples, vectorize)
    }

    public shuffle() {
        this.data = ArrayHelper.shuffle(this.data)
    }

    private loadMnist(folderPath: string, imageFileName: string, labelFileName: string, maxExamples: number, vectorize: boolean) {
        const trainFileBuffer = fs.readFileSync(path.join(folderPath + "/" + imageFileName));
        const labelFileBuffer = fs.readFileSync(path.join(folderPath + "/" + labelFileName));

        this.TOTAL_EXAMPLES = maxExamples

        let bar

        if (this.VERBOSE) {
            bar = new cliProgress.Bar({
                barCompleteChar: '#',
                barIncompleteChar: '-',
                format:'Loading mnist.. [' + '{bar}' + '] {percentage}% | {value}/{total}',
                fps: 10,
                stream: process.stdout,
                barsize: 30
            });
            bar.start(maxExamples, 0)
        }

        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            const size = 28
            const image: Tensor = new Tensor([size, size, 1], true)

            for (let x = 0; x < size; x++) {
                for (let y = 0; y < size; y++) {
                    image.t[y][x][0] = trainFileBuffer[(imageIndex * size * size) + (x + (y * size)) + 15]
                }
            }

            let exampleData: Tensor = vectorize? image.vectorize().div(255): image.div(255)

            this.DATA_SHAPE = exampleData.shape

            let example: Example = {
                data: exampleData,
                label: Tensor.toCategorical(labelFileBuffer[imageIndex + 8], 10)
            };

            this.data.push(example);
            if (this.VERBOSE) {
                bar.increment();
            }
        }
        if (this.VERBOSE) {
            bar.stop()
        }
    }

    public loadTestData(path: string, maxExamples: number = 2100) {
        const data = JSON.parse(fs.readFileSync(path, {encoding: "UTF-8"}));

        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            let example: Example = {
                data: new Tensor(data["features"][imageIndex]),
                label: Tensor.toCategorical(data["labels"][imageIndex], 3)
            };

            this.data.push(example);
        }
    }

    public getBatch(batch_id: number): Example[] {
        if (this.IS_GENERATOR) {
            return this.GENERATOR(batch_id)
        } else {
            return this.data.slice(batch_id * this.BATCH_SIZE, batch_id * this.BATCH_SIZE + this.BATCH_SIZE)
        }
    }
}
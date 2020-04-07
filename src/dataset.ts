import Vector from "./vector";
import * as fs from "fs";
import * as path from 'path';
import Jimp from 'jimp';
import Tensor from "./tensor";
import Matrix from "./matrix";

export interface Example {
    data: Vector | Matrix | Tensor,
    label: Vector
}

export default class Dataset {
    private data: Array<Example> = []

    public BATCH_SIZE = 1;
    public IS_GENERATOR = false;
    public TOTAL_EXAMPLES = 0;
    public DATA_STRUCTURE: any = undefined
    public GENERATOR: Function = () => {
    };

    public size(): number {
        return this.data.length;
    }

    public setGenerator(gen: Function) {
        this.GENERATOR = gen
    }

    public static async read_image(path: string): Promise<Tensor> {
        const image = await Jimp.read(path);
        const t = new Tensor()
        for (let i = 0; i < image.bitmap.data.length; i += 4) {
            let y = Math.floor((i / 4) / image.getHeight())
            let x = (i / 4) - (y * image.getWidth())

            for (let d = 0; d < 3; d++) {
                t.set(y, x, d, image.bitmap.data[i + d])
            }
        }
        return t
    }

    public vectorize_image(image: Tensor): Vector {
        const v = new Vector(image.count())
        let index = 0;
        image.iterate((i: number, j: number, k: number) => {
            v.set(index, image.get(i, j, k))
            index += 1
        })
        this.DATA_STRUCTURE = Vector
        return v
    }

    public loadMnistTrain(folderPath: string, maxExamples: number = 60000, vectorize: boolean = true) {
        this.loadMnist(folderPath, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", maxExamples, vectorize)
    }

    public loadMnistTest(folderPath: string, maxExamples: number = 60000, vectorize: boolean = true) {
        this.loadMnist(folderPath, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", maxExamples, vectorize)
    }

    private loadMnist(folderPath: string, imageFileName: string, labelFileName: string, maxExamples: number, vectorize: boolean) {
        const trainFileBuffer = fs.readFileSync(path.join(folderPath + "/" + imageFileName));
        const labelFileBuffer = fs.readFileSync(path.join(folderPath + "/" + labelFileName));

        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            const image: Tensor = new Tensor()
            const size = 28
            image.createEmptyArray(size, size, vectorize? 1: 3)

            for (let x = 0; x < size; x++) {
                for (let y = 0; y < size; y++) {
                    const val = trainFileBuffer[(imageIndex * size * size) + (x + (y * size)) + 15]
                    if (isNaN(val)) {
                        console.log("Failes", val)
                    }
                    image.set(y, x, 0, val)
                    if (!vectorize) {
                        image.set(y, x, 1, val)
                        image.set(y, x, 2, val)
                    }
                }
            }

            let exampleData: Tensor | Vector
            if (vectorize) {
                exampleData = this.vectorize_image(image)
            } else {
                exampleData = image
                this.DATA_STRUCTURE = Tensor
            }

            exampleData = exampleData.div(255)

            let example: Example = {
                data: exampleData,
                label: Vector.toCategorical(labelFileBuffer[imageIndex + 8], 10)
            };

            this.data.push(example);
        }
    }

    public loadTestData(path: string, maxExamples: number = 2100) {
        const data = JSON.parse(fs.readFileSync(path, {encoding: "UTF-8"}));

        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            let example: Example = {
                data: new Vector(data["features"][imageIndex]),
                label: Vector.toCategorical(data["labels"][imageIndex], 3)
            };

            this.data.push(example);
        }
    }

    public getBatch(batch: number): Array<Example> {
        return this.data.slice(batch * this.BATCH_SIZE, batch * this.BATCH_SIZE + this.BATCH_SIZE)
    }
}
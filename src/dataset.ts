import Vector from "./vector";
import * as fs from "fs";
import * as path from 'path';
import Jimp from 'jimp';
import Tensor from "./tensor";
import Matrix from "./matrix";

export interface Example {
    data: Vector | Matrix |Tensor,
    label: Vector
}

export default class Dataset {
    private data: Array<Example> = []

    public BATCH_SIZE = 1;
    public IS_GENERATOR = false;
    public TOTAL_EXAMPLES = 0;
    public GENERATOR: Function = () => {};

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
            let y = Math.floor((i/4) / image.getHeight())
            let x = (i/4) - (y * image.getWidth())

            for (let d = 0; d < 3; d++) {
                t.set(y,x,d, image.bitmap.data[i + d])
            }
        }
        return t
    }

    public vectorize_image(image: Tensor): Vector {
        const v = new Vector(image.count())
        image.iterate((i: number, j: number , k: number) => {
            v.set((i * image.dim().r + j * image.dim().c + k * image.dim().d), image.get(i,j,k))
        })
        return v
    }

    public loadMnistTrain(folderPath: string, maxExamples: number = 60000) {
        this.loadMnist(folderPath, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", maxExamples)
    }

    public loadMnistTest(folderPath: string, maxExamples: number = 60000) {
        this.loadMnist(folderPath, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", maxExamples)
    }

    private loadMnist(folderPath: string, imageFileName: string, labelFileName: string, maxExamples: number) {
        const trainFileBuffer  = fs.readFileSync(path.join(folderPath + "/" + imageFileName));
        const labelFileBuffer = fs.readFileSync(path.join(folderPath + "/" + labelFileName));

        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            let image: Tensor = new Tensor()
            image.createEmptyArray(28,28,1)

            for (let x = 0; x < 28; x++) {
                for (let y = 0; y < 28; y++) {
                    image.set(y,x,0, trainFileBuffer[(imageIndex * 28 * 28) + (x + (y * 28)) + 15])
                }
            }


            let exampleData = this.vectorize_image(image)
            exampleData = exampleData.div(255)

            let example: Example = {
                data: exampleData,
                label: Vector.toCategorical(labelFileBuffer[imageIndex + 8], 10)
            };

            this.data.push(example);
        }
    }

    public loadTestData(path: string, maxExamples: number = 2100) {
        const data  = JSON.parse(fs.readFileSync(path, {encoding:"UTF-8"}));

        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            let example: Example = {
                data: new Vector(data["features"][imageIndex]),
                label: Vector.toCategorical(data["labels"][imageIndex], 3)
            };

            this.data.push(example);
        }
    }

    public getBatch(batch: number): Array<Example> {
        return this.data.slice(batch*this.BATCH_SIZE, batch*this.BATCH_SIZE + this.BATCH_SIZE)
    }
}
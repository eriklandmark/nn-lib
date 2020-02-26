import Vector from "./vector";
import * as fs from "fs";
import * as path from 'path';

export interface Example {
    data: Vector,
    label: Vector
}

export default class Dataset {
    private data: Array<Example> = []
    public BATCH_SIZE = 1;

    public size(): number {
        return this.data.length;
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
            let pixels: number[] = [];
            for (let x = 0; x < 28; x++) {
                for (let y = 0; y < 28; y++) {
                    pixels.push(trainFileBuffer[(imageIndex * 28 * 28) + (x + (y * 28)) + 15]);
                }
            }

            let exampleData = new Vector(pixels)
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
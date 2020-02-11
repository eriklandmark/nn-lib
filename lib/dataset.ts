import Matrix from "./matrix";
import Vector from "./vector";
import * as fs from "fs";
import * as path from 'path';

export interface Example {
    data: Vector,
    label: Vector
}

export default class Dataset {
    private data: Array<Example> = []

    public loadMnist(folderPath: string, maxExamples: number = 60000) {
        const trainFileBuffer  = fs.readFileSync(path.join(folderPath + '/train-images-idx3-ubyte'));
        const labelFileBuffer = fs.readFileSync(path.join(folderPath + '/train-labels-idx1-ubyte'));

        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            let exampleData = new Vector(new Float64Array(28*28));
            let pixels: number[] = [];

            for (let x = 0; x < 28; x++) {
                for (let y = 0; y < 28; y++) {
                    pixels.push(trainFileBuffer[(imageIndex * 28 * 28) + (x + (y * 28)) + 15]);
                }
            }

            exampleData = exampleData.div(255)

            let example: Example = {
                data: exampleData,
                label: Vector.toCategorical(labelFileBuffer[imageIndex + 8], 10)
            };

            this.data.push(example);
        }
    }

    public loadMnistOld(folderPath: string, maxExamples: number = 60000) {
        const trainFileBuffer  = fs.readFileSync(path.join(folderPath + '/train-images-idx3-ubyte'));
        const labelFileBuffer = fs.readFileSync(path.join(folderPath + '/train-labels-idx1-ubyte'));

        for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
            let exampleData = new Matrix();
            exampleData.createEmptyArray(28,28)

            for (let x = 0; x < 28; x++) {
                for (let y = 0; y < 28; y++) {
                    exampleData.set(y,x, trainFileBuffer[(imageIndex * 28 * 28) + (x + (y * 28)) + 15])
                }
            }

            exampleData.div(255)

            /*let example: Example = {
                data: exampleData,
                label: Vector.toCategorical(labelFileBuffer[imageIndex + 8], 10)
            };*/

            //this.data.push(example);
        }
    }

    public getBatch(batch: number, batchSize: number): Array<Example> {
        return this.data.slice(batch*batchSize, batch*batchSize + batchSize)
    }
}
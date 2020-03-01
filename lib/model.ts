import Dataset, {Example} from "./dataset";
import OutputLayer from "./output_layer";
import DenseLayer from "./dense_layer";
import Layer from "./layer";
import * as fs from "fs";
import Matrix from "./matrix";
import Vector from "./vector";
import {GPU} from 'gpu.js';

interface SavedModel {
    layer_keys: string[],
    layers: any,
    output_layer : any
}

export default class Model {
    layers: Layer[]
    learning_rate = 0;
    gpuInstance: GPU
    USE_GPU: boolean = false;

    constructor(layers: Layer[]) {
        this.layers = layers
        this.gpuInstance = new GPU()
    }

    train_on_batch(examples: Matrix, labels: Matrix): number {
        this.layers[0].feedForward(examples)
        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].feedForward(this.layers[i - 1])
        }

        (<OutputLayer>this.layers[this.layers.length - 1]).backPropagation(labels, this.layers[this.layers.length - 2])
        for (let i = this.layers.length - 2; i > 0; i--) {
            (<DenseLayer>this.layers[i]).backPropagation(this.layers[i + 1], this.layers[i - 1])
        }
        (<DenseLayer>this.layers[0]).backPropagation(this.layers[1], examples)

        for (let layer of this.layers) {
            layer.updateWeights(this.learning_rate)
        }
        return (<OutputLayer>this.layers[this.layers.length - 1]).loss
    }

    public async train(data: Example[] | Dataset, epochs: number, learning_rate: number, shuffle: boolean = false) {
        this.learning_rate = learning_rate

        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].populate()
            this.layers[i].useGpu = this.USE_GPU
            this.layers[i].setGpuInstance(this.gpuInstance)
        }

        const shuffleArray = (array: Example[]) => {
            let currentIndex = array.length, temporaryValue, randomIndex;

            // While there remain elements to shuffle...
            while (0 !== currentIndex) {

                // Pick a remaining element...
                randomIndex = Math.floor(Math.random() * currentIndex);
                currentIndex -= 1;

                // And swap it with the current element.
                temporaryValue = array[currentIndex];
                array[currentIndex] = array[randomIndex];
                array[randomIndex] = temporaryValue;
            }

            return array;
        }

        if (data instanceof Dataset) {
            console.log("Starting training...")

            const startTime = Date.now();
            if (data.IS_GENERATOR) {
                const batch_count = Math.floor(data.TOTAL_EXAMPLES / data.BATCH_SIZE)

                console.log("Total " + batch_count + " batches for " + epochs + " epochs.")

                for (let epoch = 0; epoch < epochs; epoch++) {
                    console.log("Starting Epoch:", epoch)
                    for (let batch_id = 0; batch_id < batch_count; batch_id++) {
                        const batch = await data.GENERATOR(batch_id)
                        const examples = new Matrix(batch.map((ex: Example) => ex.data)).transpose()
                        const labels = new Matrix(batch.map((ex: Example) => ex.label)).transpose()
                        let error = this.train_on_batch(examples, labels);

                        console.log("Error for batch: " + batch_id + " =", error)
                    }

                }

            } else {
                const batch_count = Math.floor(data.size() / data.BATCH_SIZE)

                for (let epoch = 0; epoch < epochs; epoch++) {
                    console.log("Starting Epoch:", epoch)
                    for (let batch_id = 0; batch_id < batch_count; batch_id++) {
                        let batch: Example[]
                        if (shuffle) {
                            batch = shuffleArray(data.getBatch(batch_id))
                        } else {
                            batch = data.getBatch(batch_id)
                        }

                        const examples = new Matrix(batch.map((ex) => ex.data)).transpose()
                        const labels = new Matrix(batch.map((ex) => ex.label)).transpose()
                        let error = this.train_on_batch(examples, labels);

                        console.log("Error for batch: " + batch_id + " =", error)
                    }

                }
            }

            console.log("Done..")
            const duration = Math.floor((Date.now() - startTime) / 1000)
            console.log("Duration: " + duration + " seconds")
        } else {

            let examples = new Matrix(data.map((ex) => ex.data)).transpose()
            let labels = new Matrix(data.map((ex) => ex.label)).transpose()

            for (let epoch = 0; epoch < epochs; epoch++) {
                //data = shuffle(data)
                console.log(this.train_on_batch(examples, labels))
            }
        }
    }

    predict(data: Vector | Matrix): Matrix {
        let exampleMatrix: Matrix
        if (data instanceof Vector) {
            exampleMatrix = new Matrix([data]).transpose()
        } else {
            exampleMatrix = data
        }
        this.layers[0].feedForward(exampleMatrix)
        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].feedForward(this.layers[i - 1])
        }
        return this.layers[this.layers.length - 1].activation
    }

    save(path: string) {
        const modelObj: SavedModel = {
            layer_keys: [],
            layers: {},
            output_layer: {}
        }

        for (let i = 0; i < this.layers.length - 1; i++) {
            modelObj.layers[`layer_${i}`] = {}
            modelObj.layers[`layer_${i}`]["weights"] = this.layers[i].weights.matrix
            modelObj.layers[`layer_${i}`]["bias"] = this.layers[i].bias.vector
            modelObj.layer_keys.push(`layer_${i}`)
        }
        modelObj["output_layer"] = {}
        modelObj["output_layer"]["weights"] = this.layers[this.layers.length - 1].weights.matrix
        modelObj["output_layer"]["bias"] = this.layers[this.layers.length - 1].bias.vector

        fs.writeFileSync(path, JSON.stringify(modelObj))
    }

    load(path: string) {
        const modelObj = JSON.parse(fs.readFileSync(path, {encoding: "UTF-8"}))

        for (let i = 0; i < modelObj.layer_keys.length; i++) {
            const layer = modelObj.layer_keys[i]
            this.layers[i].weights = new Matrix(modelObj.layers[layer].weights.map((row: any) => {
                return Object.keys(row).map((item, index) => row[index.toString()])
            }))

            this.layers[i].bias = new Vector(Object.keys(modelObj.layers[layer].bias).map(
                (item, index) => {
                    return modelObj.layers[layer].bias[index.toString()]
                }
                ))
        }

        this.layers[this.layers.length - 1].weights = new Matrix(modelObj.output_layer.weights.map((row: any) => {
            return Object.keys(row).map((item, index) => row[index.toString()])
        }))

        this.layers[this.layers.length - 1].bias = new Vector(Object.keys(modelObj.output_layer.bias).map(
            (item, index) => {
                return modelObj.output_layer.bias[index.toString()]
            }
        ))
    }

    public isGpuAvailable():boolean {
        return GPU.isGPUSupported
    }
}
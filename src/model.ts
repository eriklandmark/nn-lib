import Dataset, {Example} from "./dataset";
import Layer from "./lib/layers/layer";
import * as fs from "fs";
import Matrix from "./matrix";
import Vector from "./vector";
import {GPU} from 'gpu.js';
import ArrayHelper from "./helpers/array_helper";
import {ILoss} from "./lib/losses/losses";
import Tensor from "./tensor";
import {LayerHelper} from "./lib/layers/layer_helper";
import Helper from "./helpers/helper";
import OutputLayer from "./lib/layers/output_layer";

export interface SavedLayer {
    weights?: Float32Array[]
    bias?: Float32Array
    shape?: number[]
    filters?: Float32Array[][][]
    nr_filters?: number
    filterSize?: number[]
    activation?: string
    loss?: string
    rate?: number
    prevLayerShape?: number[]
}

export default class Model {
    layers: Layer[]
    learning_rate = 0;
    gpuInstance: GPU
    USE_GPU: boolean = false;
    input_shape: number[] = []
    private isBuilt = false;

    constructor(layers: Layer[]) {
        this.layers = layers
        this.gpuInstance = new GPU()
    }

    public isGpuAvailable(): boolean {
        return GPU.isGPUSupported
    }

    public build(inputShape: number[], lossFunction: ILoss, verbose = true) {
        if (!(this.layers[this.layers.length - 1] instanceof OutputLayer)) {
            throw "Last layer must be an OutputLayer!..."
        }

        this.input_shape = inputShape
        this.layers[0].setGpuInstance(this.gpuInstance)
        this.layers[0].useGpu = this.USE_GPU
        this.layers[0].buildLayer(inputShape)

        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].setGpuInstance(this.gpuInstance)
            this.layers[i].useGpu = this.USE_GPU
            if (i == this.layers.length - 1) {
                (<OutputLayer>this.layers[i]).lossFunction = lossFunction
            }
            this.layers[i].buildLayer(this.layers[i - 1].shape)
        }

        if (verbose) {
            console.log("Successfully build model!")
        }

        this.isBuilt = true;
    }

    public summary() {
        if (this.isBuilt) {
            let input = {type: "input", shape: this.input_shape, activation: "NO ACTIVATION"}
            let layer_info = this.layers.map((layer) => layer.getLayerInfo())
            const sum = (acc: number, val: any) => acc + val
            let total_neurons = layer_info.map((info) => info.shape).reduce((acc, val) => {
                return acc + val.reduce(sum, 0)
            }, 0)
            console.table([input, ...layer_info])
            console.log("Total: neurons: ", total_neurons)
        } else {
            console.log("Model hasn't been built yet!..")
        }
    }

    train_on_batch(examples: Matrix | Tensor[], labels: Matrix): number {
        if (this.USE_GPU) {
            let result: any = (<Matrix>examples).toNumberArray()
            let batch_size: number = (<Matrix>examples).dim().r
            for (let i = 0; i < this.layers.length; i++) {
                this.layers[i].buildFFKernels(batch_size)
                result = this.layers[i].feedForward(result, true, true)
            }

            //@ts-ignore
            this.layers[this.layers.length - 1].backPropagationOutputLayer(labels, this.layers[this.layers.length - 2])
            this.layers[this.layers.length - 1].output_error = (<Matrix>(<OutputLayer>this.layers[this.layers.length - 1]).output_error).toNumberArray()
            for (let i = this.layers.length - 2; i >= 0; i--) {
                this.layers[i].buildBPKernels(this.layers[i + 1].weights.dim().c)
                let input: Matrix | Layer = i == 0 ? <Matrix>examples : this.layers[i - 1]
                this.layers[i].backPropagation(this.layers[i + 1], input, true)
            }

            for (let layer of this.layers) {
                layer.updateWeights(this.learning_rate)
            }

            return (<OutputLayer>this.layers[this.layers.length - 1]).loss
        } else {
            this.layers[0].feedForward(examples, true)
            for (let i = 1; i < this.layers.length; i++) {
                this.layers[i].feedForward(this.layers[i - 1], true, false)
            }

            (<OutputLayer>this.layers[this.layers.length - 1]).backPropagationOutputLayer(labels, this.layers[this.layers.length - 2])
            for (let i = this.layers.length - 2; i > 0; i--) {
                this.layers[i].backPropagation(this.layers[i + 1], this.layers[i - 1])
            }
            this.layers[0].backPropagation(this.layers[1], examples)

            for (let layer of this.layers) {
                layer.updateWeights(this.learning_rate)
            }

            return (<OutputLayer>this.layers[this.layers.length - 1]).loss
        }
    }

    public async train(data: Example[] | Dataset, epochs: number, learning_rate: number, shuffle: boolean = false, verbose: boolean = true) {
        if (!this.isBuilt) {
            throw "Model hasn't been build yet!.."
        }

        this.learning_rate = learning_rate

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
                            batch = ArrayHelper.shuffle(data.getBatch(batch_id))
                        } else {
                            batch = data.getBatch(batch_id)
                        }

                        let examples: Matrix | Tensor[]
                        let error: number = 0
                        let exampleData = <Vector[] | Tensor[]>batch.map((ex) => ex.data)
                        const labels = new Matrix(batch.map((ex) => ex.label)).transpose()
                        if (data.DATA_STRUCTURE == Vector) {
                            examples = new Matrix(batch.map((ex) => <Vector>ex.data)).transpose()
                        } else if (data.DATA_STRUCTURE == Tensor) {
                            examples = <Tensor[]>exampleData
                        }

                        const seconds = await Helper.timeit(() => {
                            error = this.train_on_batch(examples, labels);
                        }, false)

                        console.log("Error for batch: " + batch_id + " =", error, "| Time:", seconds, "seconds")
                    }

                }
            }

            console.log("Done..")
            const duration = Math.floor((Date.now() - startTime) / 1000)
            console.log("Duration: " + duration + " seconds")
        } else {
            let exampleData = <Vector[] | Tensor[]>data.map((ex) => ex.data)
            let examples = exampleData[0] instanceof Vector ? new Matrix(<Vector[]>exampleData) : <Tensor[]>exampleData
            let labels = new Matrix(data.map((ex) => ex.label)).transpose()

            for (let epoch = 0; epoch < epochs; epoch++) {
                console.log(this.train_on_batch(examples, labels))
            }
        }
    }

    predict(data: Vector | Matrix | Tensor): Matrix {
        if (!this.isBuilt) {
            throw "Model hasn't been build yet!.."
        }
        let exampleMatrix: Matrix | Tensor[]
        if (data instanceof Vector) {
            exampleMatrix = new Matrix([data]).transpose()
        } else {
            exampleMatrix = [<Tensor>data]
        }
        this.layers[0].feedForward(exampleMatrix, false)
        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].feedForward(this.layers[i - 1], false)
        }
        return <Matrix>this.layers[this.layers.length - 1].activation
    }

    save(path: string) {
        const modelObj = {layers: {}}

        for (let i = 0; i < this.layers.length; i++) {
            modelObj.layers[`layer_${i}`] = {
                type: this.layers[i].type,
                info: this.layers[i].toSavedModel()
            }
        }

        fs.writeFileSync(path, JSON.stringify(modelObj))
    }

    load(path: string) {

        const modelObj = JSON.parse(fs.readFileSync(path, {encoding: "UTF-8"}))

        const layer_keys: string[] = Object.keys(modelObj.layers).sort()
        this.layers = []
        for (let layer_key of layer_keys) {
            let layer = LayerHelper.fromType(modelObj.layers[layer_key].type)
            layer.fromSavedModel(modelObj.layers[layer_key].info)
            this.layers.push(layer)
        }

        this.isBuilt = true


        /*
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

        if (!this.isBuilt) {
            throw "Model hasn't been build yet!.."
        }*/
    }
}
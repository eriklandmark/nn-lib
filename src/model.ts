import Dataset, {Example} from "./dataset";
import Layer from "./layers/layer";
import * as fs from "fs";
import Matrix from "./matrix";
import Vector from "./vector";
import {GPU} from 'gpu.js';
import Tensor from "./tensor";
import {LayerHelper} from "./layers/layer_helper";
import Helper from "./helpers/helper";
import OutputLayer from "./layers/output_layer";
import Path from "path";
import cliProgress from "cli-progress";
import StochasticGradientDescent from "./optimizers/StochasticGradientDescent";

export interface SavedLayer {
    weights: Float32Array[] | Float32Array[][][]
    bias: Float32Array | Float32Array[]
    shape: number[]
    activation: string
    prevLayerShape: number[]
    optimizer: string
    layer_specific: {
        nr_filters?: number
        filterSize?: number[]
        stride?: number[] | number
        padding?: number
        poolingFunc?: string
        loss?: string
        rate?: number
        [propName: string]: any
    }
}

interface ModelSettings {
    USE_GPU: boolean,
    BACKLOG: boolean,
    SAVE_CHECKPOINTS: boolean,
    MODEL_SAVE_PATH: string,
    VERBOSE_COMPACT: boolean
}

export interface BacklogData {
    actual_duration: number,
    calculated_duration: number
    epochs: {
        [propName: string]: {
            total_loss: number,
            total_accuracy: number,
            batches: {accuracy: number, loss: number, time: number}[],
            calculated_duration: number,
            actual_duration: number
        };
    }
}

export default class Model {
    layers: Layer[]
    gpuInstance: GPU
    private isBuilt = false;
    backlog: BacklogData = {
        actual_duration: 0, calculated_duration: 0, epochs:{}
    }
    settings: ModelSettings = {
        USE_GPU: false,
        BACKLOG: true,
        SAVE_CHECKPOINTS: false,
        MODEL_SAVE_PATH: "",
        VERBOSE_COMPACT: true
    }
    model_data: {
        input_shape: number[]
        learning_rate: number
        last_epoch: number
    } = {
        input_shape: [0],
        learning_rate: 0,
        last_epoch: 0
    }

    constructor(layers: Layer[]) {
        this.layers = layers
        this.gpuInstance = new GPU()
    }

    public isGpuAvailable(): boolean {
        return GPU.isGPUSupported
    }

    public build(inputShape: number[], learning_rate: number, lossFunction,
                 optimizer = StochasticGradientDescent, verbose: boolean = true) {
        if (!(this.layers[this.layers.length - 1] instanceof OutputLayer)) {
            throw "Last layer must be an OutputLayer!..."
        }

        if(!this.isGpuAvailable() && this.settings.USE_GPU) {
            console.error("GPU is not supported.. falling back on CPU.")
            this.settings.USE_GPU = false
        }

        if(this.settings.SAVE_CHECKPOINTS && !this.settings.MODEL_SAVE_PATH) {
            console.error("No model path specified.. Turning of saving checkpoints.")
            this.settings.SAVE_CHECKPOINTS = false
        }

        if(this.settings.BACKLOG && !this.settings.MODEL_SAVE_PATH) {
            console.error("No model path specified.. Turning of saving backlog.")
            this.settings.SAVE_CHECKPOINTS = false
        }

        if(this.settings.MODEL_SAVE_PATH) {
            if (!fs.existsSync(this.settings.MODEL_SAVE_PATH)) {
                fs.mkdirSync(this.settings.MODEL_SAVE_PATH)
            }
        }

        this.model_data.learning_rate = learning_rate
        this.model_data.input_shape = inputShape
        this.layers[0].isFirstLayer = true

        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].gpuInstance = this.gpuInstance
            this.layers[i].useGpu = this.settings.USE_GPU
            this.layers[i].learning_rate = learning_rate
            if (i == this.layers.length - 1) {
                (<OutputLayer>this.layers[i]).lossFunction = new lossFunction()
            }
            this.layers[i].buildLayer(i == 0? inputShape: this.layers[i - 1].shape)
            this.layers[i].optimizer = new optimizer(this.layers[i])
        }

        if (verbose) {
            console.log("Successfully build model!")
        }

        this.isBuilt = true;
    }

    public summary() {
        if (this.isBuilt) {
            let input = {type: "input", shape: this.model_data.input_shape, activation: "NONE"}
            let layer_info = this.layers.map((layer) => layer.getLayerInfo())
            let total_neurons = layer_info.map((info) => info.shape).reduce((acc, val) => {
                return acc + val.reduce((a, s) => a*s, 1)
            }, 0)
            console.table([input, ...layer_info])
            console.log("Total: neurons: ", total_neurons)
        } else {
            console.log("Model hasn't been built yet!..")
        }
    }

    train_on_batch(examples: Matrix | Tensor[], labels: Matrix): any {
        if (this.settings.USE_GPU) {
            let result: any = examples instanceof Matrix?(<Matrix>examples).toNumberArray():
                examples.map((t) => t.toNumberArray())
            let batch_size: number = examples instanceof Matrix? (<Matrix>examples).dim().r:
                examples.length
            for (let i = 0; i < this.layers.length; i++) {
                if (this.layers[i].hasGPUSupport) {
                    this.layers[i].buildFFKernels(batch_size)
                    result = this.layers[i].feedForward(result, true)
                } else {
                    this.layers[i].feedForward(i == 0? examples : this.layers[i - 1], true)
                }
            }

            //@ts-ignore
            this.layers[this.layers.length - 1].backPropagationOutputLayer(labels, this.layers[this.layers.length - 2])
            this.layers[this.layers.length - 1].output_error = (<Matrix>(<OutputLayer>this.layers[this.layers.length - 1]).output_error).toNumberArray()
            for (let i = this.layers.length - 2; i >= 0; i--) {
                if (this.layers[i].hasGPUSupport) {
                    //this.layers[i].buildBPKernels(this.layers[i + 1].weights.dim().c)
                }
                let input: Matrix | Layer = i == 0 ? <Matrix>examples : this.layers[i - 1]
                this.layers[i].backPropagation(this.layers[i + 1], input)
            }

            for (let layer of this.layers) {
                layer.updateLayer()
            }

            return {loss:(<OutputLayer>this.layers[this.layers.length - 1]).loss,
                    accuracy: (<OutputLayer>this.layers[this.layers.length - 1]).accuracy}
        } else {
            this.layers[0].feedForward(examples, true)
            for (let i = 1; i < this.layers.length; i++) {
                this.layers[i].feedForward(this.layers[i - 1], true)
            }

            (<OutputLayer>this.layers[this.layers.length - 1]).backPropagationOutputLayer(labels, this.layers[this.layers.length - 2])
            for (let i = this.layers.length - 2; i > 0; i--) {
                this.layers[i].backPropagation(this.layers[i + 1], this.layers[i - 1])
            }
            this.layers[0].backPropagation(this.layers[1], examples)

            for (let layer of this.layers) {
                layer.updateLayer()
            }

            return {loss:(<OutputLayer>this.layers[this.layers.length - 1]).loss,
                accuracy: (<OutputLayer>this.layers[this.layers.length - 1]).accuracy}
        }
    }

    public async train(data: Example[] | Dataset, epochs: number, shuffle: boolean = false) {
        if (!this.isBuilt) {
            throw "Model hasn't been build yet!.."
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

                for (let epoch = 1; epoch <= epochs; epoch++) {
                    console.log("----------------")
                    console.log("Starting Epoch:", epoch, "/", epochs)
                    if (shuffle) {
                        data.shuffle()
                    }
                    const epoch_data = {
                        total_loss: 0,
                        total_accuracy: 0,
                        batches: [],
                        calculated_duration: 0,
                        actual_duration: 0
                    }
                    const epoch_startTime =  Date.now()

                    const bar = new cliProgress.Bar({
                        barCompleteChar: '#',
                        barIncompleteChar: '-',
                        format:'Batch: {value}/{total} [' + '{bar}' + '] {percentage}% | loss: {loss} | acc: {acc} | Time (TOT/AVG): {time_tot} / {time_avg}',
                        fps: 10,
                        stream: process.stdout,
                        barsize: 15
                    });

                    if(this.settings.VERBOSE_COMPACT) {
                        bar.start(batch_count, 0, {
                            acc: (0).toPrecision(3),
                            time_tot: (0).toPrecision(5),
                            time_avg: (0).toPrecision(5),
                            loss: (0).toPrecision(5)
                        })
                    }

                    for (let batch_id = 0; batch_id < batch_count; batch_id++) {
                        let batch: Example[] = data.getBatch(batch_id)

                        let examples: Matrix | Tensor[]
                        let b_loss: number = 0
                        let b_acc: number = 0
                        let exampleData = <Vector[] | Tensor[]>batch.map((ex) => ex.data)
                        const labels = new Matrix(batch.map((ex) => ex.label)).transpose()
                        if (data.DATA_STRUCTURE == Vector) {
                            examples = new Matrix(batch.map((ex) => <Vector>ex.data)).transpose()
                        } else if (data.DATA_STRUCTURE == Tensor) {
                            examples = <Tensor[]>exampleData
                        }

                        const seconds = await Helper.timeit(() => {
                            let {loss, accuracy} = this.train_on_batch(examples, labels);
                            b_loss = loss
                            b_acc = accuracy
                        }, false)
                        epoch_data.batches.push({accuracy: b_acc, loss: b_loss, time: seconds})
                        epoch_data.total_loss += b_loss
                        epoch_data.total_accuracy += b_acc
                        epoch_data.calculated_duration += seconds
                        this.backlog.calculated_duration += seconds
                        this.backlog.epochs["epoch_" + epoch] = epoch_data
                        this.saveBacklog()

                        if (this.settings.VERBOSE_COMPACT) {
                            bar.increment(1, {
                                acc: (epoch_data.total_accuracy/(batch_id + 1)).toPrecision(3),
                                time_tot: epoch_data.calculated_duration.toPrecision(5),
                                time_avg: (epoch_data.calculated_duration/(batch_id + 1)).toPrecision(4),
                                loss: (epoch_data.total_loss/(batch_id + 1)).toPrecision(5)
                            })
                        } else {
                            console.log("Batch:", (batch_id + 1), "/", batch_count,
                                "Loss =", b_loss,", Acc = ", b_acc , "| Time:", seconds, "seconds")
                        }
                    }
                    bar.stop()

                    epoch_data.actual_duration = (Date.now() - epoch_startTime) / 1000
                    this.backlog.epochs["epoch_" + epoch] = epoch_data
                    console.log("Loss: TOT", epoch_data.total_loss.toPrecision(5),
                        "AVG",(epoch_data.total_loss / batch_count).toPrecision(5),
                        "| Accuracy:", (epoch_data.total_accuracy / batch_count).toPrecision(3),
                        "| Total time:", epoch_data.actual_duration.toPrecision(5), "/",
                        epoch_data.calculated_duration.toPrecision(4))
                    this.saveBacklog()
                    this.model_data.last_epoch = epoch
                    if (this.settings.SAVE_CHECKPOINTS) {
                        this.save("model_checkpoint_" + epoch + ".json")
                    }
                }
            }

            console.log("Done..")
            const duration = (Date.now() - startTime) / 1000
            this.backlog.actual_duration = duration
            console.log("Duration: " + duration + " seconds")
            this.saveBacklog()
        } else {
            let exampleData = <Vector[] | Tensor[]>data.map((ex) => ex.data)
            let examples = exampleData[0] instanceof Vector ? new Matrix(<Vector[]>exampleData) : <Tensor[]>exampleData
            let labels = new Matrix(data.map((ex) => ex.label)).transpose()

            for (let epoch = 0; epoch < epochs; epoch++) {
                console.log(this.train_on_batch(examples, labels))
            }
        }
    }

    saveBacklog() {
        if (this.settings.BACKLOG) {
            const path = Path.join(this.settings.MODEL_SAVE_PATH, "backlog.json")
            fs.writeFileSync(path, JSON.stringify(this.backlog))
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

    save(model_path: string = "model.json") {
        const modelObj = {
            model_data: this.model_data,
            settings: this.settings,
            layers: {}
        }

        for (let i = 0; i < this.layers.length; i++) {
            modelObj.layers[`layer_${i}`] = {
                type: this.layers[i].type,
                info: this.layers[i].toSavedModel()
            }
        }

        const path = Path.join(this.settings.MODEL_SAVE_PATH, model_path)
        fs.writeFileSync(path, JSON.stringify(modelObj))
    }

    load(path: string, verbose:boolean = true) {
        if(!fs.existsSync(path)) {
            throw "Model file not found!!"
        }
        const modelObj = JSON.parse(fs.readFileSync(path, {encoding: "UTF-8"}))
        this.model_data = modelObj.model_data
        this.settings = modelObj.settings
        const layer_keys: string[] = Object.keys(modelObj.layers).sort()
        this.layers = []

        if(!this.isGpuAvailable() && this.settings.USE_GPU) {
            console.error("GPU is not supported.. falling back on CPU.")
            this.settings.USE_GPU = false
        }

        for (let layer_key of layer_keys) {
            let layer = LayerHelper.fromType(modelObj.layers[layer_key].type)
            layer.fromSavedModel(modelObj.layers[layer_key].info)
            layer.gpuInstance = this.gpuInstance
            layer.useGpu = this.settings.USE_GPU
            layer.learning_rate = this.model_data.learning_rate
            this.layers.push(layer)
        }
        this.layers[0].isFirstLayer = true
        this.isBuilt = true

        if (verbose) {
            console.log("Successfully build model!")
        }
    }
}
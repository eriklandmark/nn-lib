"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (Object.hasOwnProperty.call(mod, k)) result[k] = mod[k];
    result["default"] = mod;
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
const dataset_1 = __importDefault(require("./dataset"));
const fs = __importStar(require("fs"));
const matrix_1 = __importDefault(require("./matrix"));
const vector_1 = __importDefault(require("./vector"));
const gpu_js_1 = require("gpu.js");
const tensor_1 = __importDefault(require("./tensor"));
const layer_helper_1 = require("./layers/layer_helper");
const helper_1 = __importDefault(require("./helpers/helper"));
const output_layer_1 = __importDefault(require("./layers/output_layer"));
const path_1 = __importDefault(require("path"));
const cli_progress_1 = __importDefault(require("cli-progress"));
const StochasticGradientDescent_1 = __importDefault(require("./optimizers/StochasticGradientDescent"));
class Model {
    constructor(layers) {
        this.isBuilt = false;
        this.backlog = {
            actual_duration: 0, calculated_duration: 0, epochs: {}
        };
        this.settings = {
            USE_GPU: false,
            BACKLOG: true,
            SAVE_CHECKPOINTS: false,
            MODEL_SAVE_PATH: "",
            VERBOSE_COMPACT: true
        };
        this.model_data = {
            input_shape: [0],
            learning_rate: 0,
            last_epoch: 0
        };
        this.layers = layers;
        this.gpuInstance = new gpu_js_1.GPU();
    }
    isGpuAvailable() {
        return gpu_js_1.GPU.isGPUSupported;
    }
    build(inputShape, learning_rate, lossFunction, optimizer = StochasticGradientDescent_1.default, verbose = true) {
        if (!(this.layers[this.layers.length - 1] instanceof output_layer_1.default)) {
            throw "Last layer must be an OutputLayer!...";
        }
        if (!this.isGpuAvailable() && this.settings.USE_GPU) {
            console.error("GPU is not supported.. falling back on CPU.");
            this.settings.USE_GPU = false;
        }
        if (this.settings.SAVE_CHECKPOINTS && !this.settings.MODEL_SAVE_PATH) {
            console.error("No model path specified.. Turning of saving checkpoints.");
            this.settings.SAVE_CHECKPOINTS = false;
        }
        if (this.settings.BACKLOG && !this.settings.MODEL_SAVE_PATH) {
            console.error("No model path specified.. Turning of saving backlog.");
            this.settings.SAVE_CHECKPOINTS = false;
        }
        if (this.settings.MODEL_SAVE_PATH) {
            if (!fs.existsSync(this.settings.MODEL_SAVE_PATH)) {
                fs.mkdirSync(this.settings.MODEL_SAVE_PATH);
            }
        }
        this.model_data.learning_rate = learning_rate;
        this.model_data.input_shape = inputShape;
        this.layers[0].isFirstLayer = true;
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].gpuInstance = this.gpuInstance;
            this.layers[i].useGpu = this.settings.USE_GPU;
            this.layers[i].learning_rate = learning_rate;
            if (i == this.layers.length - 1) {
                this.layers[i].lossFunction = new lossFunction();
            }
            this.layers[i].buildLayer(i == 0 ? inputShape : this.layers[i - 1].shape);
            this.layers[i].optimizer = new optimizer(this.layers[i]);
        }
        if (verbose) {
            console.log("Successfully build model!");
        }
        this.isBuilt = true;
    }
    summary() {
        if (this.isBuilt) {
            let input = { type: "input", shape: this.model_data.input_shape, activation: "NONE" };
            let layer_info = this.layers.map((layer) => layer.getLayerInfo());
            let total_neurons = layer_info.map((info) => info.shape).reduce((acc, val) => {
                return acc + val.reduce((a, s) => a * s, 1);
            }, 0);
            console.table([input, ...layer_info]);
            console.log("Total: neurons: ", total_neurons);
        }
        else {
            console.log("Model hasn't been built yet!..");
        }
    }
    train_on_batch(examples, labels) {
        if (this.settings.USE_GPU) {
            let result = examples instanceof matrix_1.default ? examples.toNumberArray() :
                examples.map((t) => t.toNumberArray());
            let batch_size = examples instanceof matrix_1.default ? examples.dim().r :
                examples.length;
            for (let i = 0; i < this.layers.length; i++) {
                if (this.layers[i].hasGPUSupport) {
                    this.layers[i].buildFFKernels(batch_size);
                    result = this.layers[i].feedForward(result, true);
                }
                else {
                    this.layers[i].feedForward(i == 0 ? examples : this.layers[i - 1], true);
                }
            }
            //@ts-ignore
            this.layers[this.layers.length - 1].backPropagationOutputLayer(labels, this.layers[this.layers.length - 2]);
            this.layers[this.layers.length - 1].output_error = this.layers[this.layers.length - 1].output_error.toNumberArray();
            for (let i = this.layers.length - 2; i >= 0; i--) {
                if (this.layers[i].hasGPUSupport) {
                    //this.layers[i].buildBPKernels(this.layers[i + 1].weights.dim().c)
                }
                let input = i == 0 ? examples : this.layers[i - 1];
                this.layers[i].backPropagation(this.layers[i + 1], input);
            }
            for (let layer of this.layers) {
                layer.updateLayer();
            }
            return { loss: this.layers[this.layers.length - 1].loss,
                accuracy: this.layers[this.layers.length - 1].accuracy };
        }
        else {
            this.layers[0].feedForward(examples, true);
            for (let i = 1; i < this.layers.length; i++) {
                this.layers[i].feedForward(this.layers[i - 1], true);
            }
            this.layers[this.layers.length - 1].backPropagationOutputLayer(labels, this.layers[this.layers.length - 2]);
            for (let i = this.layers.length - 2; i > 0; i--) {
                this.layers[i].backPropagation(this.layers[i + 1], this.layers[i - 1]);
            }
            this.layers[0].backPropagation(this.layers[1], examples);
            for (let layer of this.layers) {
                layer.updateLayer();
            }
            return { loss: this.layers[this.layers.length - 1].loss,
                accuracy: this.layers[this.layers.length - 1].accuracy };
        }
    }
    train(data, epochs, shuffle = false) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.isBuilt) {
                throw "Model hasn't been build yet!..";
            }
            if (data instanceof dataset_1.default) {
                console.log("Starting training...");
                const startTime = Date.now();
                if (data.IS_GENERATOR) {
                    const batch_count = Math.floor(data.TOTAL_EXAMPLES / data.BATCH_SIZE);
                    console.log("Total " + batch_count + " batches for " + epochs + " epochs.");
                    for (let epoch = 0; epoch < epochs; epoch++) {
                        console.log("Starting Epoch:", epoch);
                        for (let batch_id = 0; batch_id < batch_count; batch_id++) {
                            const batch = yield data.GENERATOR(batch_id);
                            const examples = new matrix_1.default(batch.map((ex) => ex.data)).transpose();
                            const labels = new matrix_1.default(batch.map((ex) => ex.label)).transpose();
                            let error = this.train_on_batch(examples, labels);
                            console.log("Error for batch: " + batch_id + " =", error);
                        }
                    }
                }
                else {
                    const batch_count = Math.floor(data.size() / data.BATCH_SIZE);
                    for (let epoch = 1; epoch <= epochs; epoch++) {
                        console.log("----------------");
                        console.log("Starting Epoch:", epoch, "/", epochs);
                        if (shuffle) {
                            data.shuffle();
                        }
                        const epoch_data = {
                            total_loss: 0,
                            total_accuracy: 0,
                            batches: [],
                            calculated_duration: 0,
                            actual_duration: 0
                        };
                        const epoch_startTime = Date.now();
                        const bar = new cli_progress_1.default.Bar({
                            barCompleteChar: '#',
                            barIncompleteChar: '-',
                            format: 'Batch: {value}/{total} [' + '{bar}' + '] {percentage}% | loss: {loss} | acc: {acc} | Time (TOT/AVG): {time_tot} / {time_avg}',
                            fps: 10,
                            stream: process.stdout,
                            barsize: 15
                        });
                        if (this.settings.VERBOSE_COMPACT) {
                            bar.start(batch_count, 0, {
                                acc: (0).toPrecision(3),
                                time_tot: (0).toPrecision(5),
                                time_avg: (0).toPrecision(5),
                                loss: (0).toPrecision(5)
                            });
                        }
                        for (let batch_id = 0; batch_id < batch_count; batch_id++) {
                            let batch = data.getBatch(batch_id);
                            let examples;
                            let b_loss = 0;
                            let b_acc = 0;
                            let exampleData = batch.map((ex) => ex.data);
                            const labels = new matrix_1.default(batch.map((ex) => ex.label)).transpose();
                            if (data.DATA_STRUCTURE == vector_1.default) {
                                examples = new matrix_1.default(batch.map((ex) => ex.data)).transpose();
                            }
                            else if (data.DATA_STRUCTURE == tensor_1.default) {
                                examples = exampleData;
                            }
                            const seconds = yield helper_1.default.timeit(() => {
                                let { loss, accuracy } = this.train_on_batch(examples, labels);
                                b_loss = loss;
                                b_acc = accuracy;
                            }, false);
                            epoch_data.batches.push({ accuracy: b_acc, loss: b_loss, time: seconds });
                            epoch_data.total_loss += b_loss;
                            epoch_data.total_accuracy += b_acc;
                            epoch_data.calculated_duration += seconds;
                            this.backlog.calculated_duration += seconds;
                            this.backlog.epochs["epoch_" + epoch] = epoch_data;
                            this.saveBacklog();
                            if (this.settings.VERBOSE_COMPACT) {
                                bar.increment(1, {
                                    acc: (epoch_data.total_accuracy / (batch_id + 1)).toPrecision(3),
                                    time_tot: epoch_data.calculated_duration.toPrecision(5),
                                    time_avg: (epoch_data.calculated_duration / (batch_id + 1)).toPrecision(4),
                                    loss: (epoch_data.total_loss / (batch_id + 1)).toPrecision(5)
                                });
                            }
                            else {
                                console.log("Batch:", (batch_id + 1), "/", batch_count, "Loss =", b_loss, ", Acc = ", b_acc, "| Time:", seconds, "seconds");
                            }
                        }
                        bar.stop();
                        epoch_data.actual_duration = (Date.now() - epoch_startTime) / 1000;
                        this.backlog.epochs["epoch_" + epoch] = epoch_data;
                        console.log("Loss: TOT", epoch_data.total_loss.toPrecision(5), "AVG", (epoch_data.total_loss / batch_count).toPrecision(5), "| Accuracy:", (epoch_data.total_accuracy / batch_count).toPrecision(3), "| Total time:", epoch_data.actual_duration.toPrecision(5), "/", epoch_data.calculated_duration.toPrecision(4));
                        this.saveBacklog();
                        this.model_data.last_epoch = epoch;
                        if (this.settings.SAVE_CHECKPOINTS) {
                            this.save("model_checkpoint_" + epoch + ".json");
                        }
                    }
                }
                console.log("Done..");
                const duration = (Date.now() - startTime) / 1000;
                this.backlog.actual_duration = duration;
                console.log("Duration: " + duration + " seconds");
                this.saveBacklog();
            }
            else {
                let exampleData = data.map((ex) => ex.data);
                let examples = exampleData[0] instanceof vector_1.default ? new matrix_1.default(exampleData) : exampleData;
                let labels = new matrix_1.default(data.map((ex) => ex.label)).transpose();
                for (let epoch = 0; epoch < epochs; epoch++) {
                    console.log(this.train_on_batch(examples, labels));
                }
            }
        });
    }
    saveBacklog() {
        if (this.settings.BACKLOG) {
            const path = path_1.default.join(this.settings.MODEL_SAVE_PATH, "backlog.json");
            fs.writeFileSync(path, JSON.stringify(this.backlog));
        }
    }
    predict(data) {
        if (!this.isBuilt) {
            throw "Model hasn't been build yet!..";
        }
        let exampleMatrix;
        if (data instanceof vector_1.default) {
            exampleMatrix = new matrix_1.default([data]).transpose();
        }
        else {
            exampleMatrix = [data];
        }
        this.layers[0].feedForward(exampleMatrix, false);
        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i].feedForward(this.layers[i - 1], false);
        }
        return this.layers[this.layers.length - 1].activation;
    }
    save(model_path = "model.json") {
        const modelObj = {
            model_data: this.model_data,
            settings: this.settings,
            layers: {}
        };
        for (let i = 0; i < this.layers.length; i++) {
            modelObj.layers[`layer_${i}`] = {
                type: this.layers[i].type,
                info: this.layers[i].toSavedModel()
            };
        }
        const path = path_1.default.join(this.settings.MODEL_SAVE_PATH, model_path);
        fs.writeFileSync(path, JSON.stringify(modelObj));
    }
    load(path, verbose = true) {
        if (!fs.existsSync(path)) {
            throw "Model file not found!!";
        }
        const modelObj = JSON.parse(fs.readFileSync(path, { encoding: "UTF-8" }));
        this.model_data = modelObj.model_data;
        this.settings = modelObj.settings;
        const layer_keys = Object.keys(modelObj.layers).sort();
        this.layers = [];
        if (!this.isGpuAvailable() && this.settings.USE_GPU) {
            console.error("GPU is not supported.. falling back on CPU.");
            this.settings.USE_GPU = false;
        }
        for (let layer_key of layer_keys) {
            let layer = layer_helper_1.LayerHelper.fromType(modelObj.layers[layer_key].type);
            layer.fromSavedModel(modelObj.layers[layer_key].info);
            layer.gpuInstance = this.gpuInstance;
            layer.useGpu = this.settings.USE_GPU;
            layer.learning_rate = this.model_data.learning_rate;
            this.layers.push(layer);
        }
        this.layers[0].isFirstLayer = true;
        this.isBuilt = true;
        if (verbose) {
            console.log("Successfully build model!");
        }
    }
}
exports.default = Model;

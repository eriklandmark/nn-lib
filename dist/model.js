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
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
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
var dataset_1 = __importDefault(require("./dataset"));
var output_layer_1 = __importDefault(require("./lib/layers/output_layer"));
var fs = __importStar(require("fs"));
var matrix_1 = __importDefault(require("./matrix"));
var vector_1 = __importDefault(require("./vector"));
var gpu_js_1 = require("gpu.js");
var array_helper_1 = __importDefault(require("./helpers/array_helper"));
var tensor_1 = __importDefault(require("./tensor"));
var layer_helper_1 = require("./lib/layers/layer_helper");
var Model = /** @class */ (function () {
    function Model(layers) {
        this.learning_rate = 0;
        this.USE_GPU = false;
        this.isBuilt = false;
        this.layers = layers;
        this.gpuInstance = new gpu_js_1.GPU();
    }
    Model.prototype.isGpuAvailable = function () {
        return gpu_js_1.GPU.isGPUSupported;
    };
    Model.prototype.build = function (inputShape, lossFunction, verbose) {
        if (verbose === void 0) { verbose = true; }
        this.layers[0].buildLayer(inputShape);
        this.layers[0].useGpu = this.USE_GPU;
        this.layers[0].setGpuInstance(this.gpuInstance);
        for (var i = 1; i < this.layers.length; i++) {
            this.layers[i].buildLayer(this.layers[i - 1].shape);
            this.layers[i].useGpu = this.USE_GPU;
            this.layers[i].setGpuInstance(this.gpuInstance);
        }
        var lastLayer = this.layers[this.layers.length - 1];
        if (lastLayer instanceof output_layer_1.default) {
            lastLayer.lossFunction = lossFunction;
        }
        else {
            throw "Last layer must be an OutputLayer!...";
        }
        if (verbose) {
            console.log("Successfully build model!");
        }
        this.isBuilt = true;
    };
    Model.prototype.summary = function () {
        if (this.isBuilt) {
            var layer_info = this.layers.map(function (layer) { return layer.getLayerInfo(); });
            console.table(layer_info);
        }
        else {
            console.log("Model hasn't been built yet!..");
        }
    };
    Model.prototype.train_on_batch = function (examples, labels) {
        this.layers[0].feedForward(examples, true);
        for (var i = 1; i < this.layers.length; i++) {
            this.layers[i].feedForward(this.layers[i - 1], true);
        }
        this.layers[this.layers.length - 1].backPropagationOutputLayer(labels, this.layers[this.layers.length - 2]);
        for (var i = this.layers.length - 2; i > 0; i--) {
            this.layers[i].backPropagation(this.layers[i + 1], this.layers[i - 1]);
        }
        this.layers[0].backPropagation(this.layers[1], examples);
        for (var _i = 0, _a = this.layers; _i < _a.length; _i++) {
            var layer = _a[_i];
            layer.updateWeights(this.learning_rate);
        }
        return this.layers[this.layers.length - 1].loss;
    };
    Model.prototype.train = function (data, epochs, learning_rate, shuffle, verbose) {
        if (shuffle === void 0) { shuffle = false; }
        if (verbose === void 0) { verbose = true; }
        return __awaiter(this, void 0, void 0, function () {
            var startTime, batch_count, epoch, batch_id, batch, examples, labels, error, batch_count, epoch, batch_id, batch, examples, error, exampleData, labels, duration, exampleData, examples, labels, epoch;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!this.isBuilt) {
                            throw "Model hasn't been build yet!..";
                        }
                        this.learning_rate = learning_rate;
                        if (!(data instanceof dataset_1.default)) return [3 /*break*/, 9];
                        console.log("Starting training...");
                        startTime = Date.now();
                        if (!data.IS_GENERATOR) return [3 /*break*/, 7];
                        batch_count = Math.floor(data.TOTAL_EXAMPLES / data.BATCH_SIZE);
                        console.log("Total " + batch_count + " batches for " + epochs + " epochs.");
                        epoch = 0;
                        _a.label = 1;
                    case 1:
                        if (!(epoch < epochs)) return [3 /*break*/, 6];
                        console.log("Starting Epoch:", epoch);
                        batch_id = 0;
                        _a.label = 2;
                    case 2:
                        if (!(batch_id < batch_count)) return [3 /*break*/, 5];
                        return [4 /*yield*/, data.GENERATOR(batch_id)];
                    case 3:
                        batch = _a.sent();
                        examples = new matrix_1.default(batch.map(function (ex) { return ex.data; })).transpose();
                        labels = new matrix_1.default(batch.map(function (ex) { return ex.label; })).transpose();
                        error = this.train_on_batch(examples, labels);
                        console.log("Error for batch: " + batch_id + " =", error);
                        _a.label = 4;
                    case 4:
                        batch_id++;
                        return [3 /*break*/, 2];
                    case 5:
                        epoch++;
                        return [3 /*break*/, 1];
                    case 6: return [3 /*break*/, 8];
                    case 7:
                        batch_count = Math.floor(data.size() / data.BATCH_SIZE);
                        for (epoch = 0; epoch < epochs; epoch++) {
                            console.log("Starting Epoch:", epoch);
                            for (batch_id = 0; batch_id < batch_count; batch_id++) {
                                batch = void 0;
                                if (shuffle) {
                                    batch = array_helper_1.default.shuffle(data.getBatch(batch_id));
                                }
                                else {
                                    batch = data.getBatch(batch_id);
                                }
                                examples = void 0;
                                error = 0;
                                exampleData = batch.map(function (ex) { return ex.data; });
                                labels = new matrix_1.default(batch.map(function (ex) { return ex.label; })).transpose();
                                if (data.DATA_STRUCTURE == vector_1.default) {
                                    examples = new matrix_1.default(batch.map(function (ex) { return ex.data; })).transpose();
                                    error = this.train_on_batch(examples, labels);
                                }
                                else if (data.DATA_STRUCTURE == tensor_1.default) {
                                    examples = exampleData;
                                }
                                error = this.train_on_batch(examples, labels);
                                console.log("Error for batch: " + batch_id + " =", error);
                            }
                        }
                        _a.label = 8;
                    case 8:
                        console.log("Done..");
                        duration = Math.floor((Date.now() - startTime) / 1000);
                        console.log("Duration: " + duration + " seconds");
                        return [3 /*break*/, 10];
                    case 9:
                        exampleData = data.map(function (ex) { return ex.data; });
                        examples = exampleData[0] instanceof vector_1.default ? new matrix_1.default(exampleData) : exampleData;
                        labels = new matrix_1.default(data.map(function (ex) { return ex.label; })).transpose();
                        for (epoch = 0; epoch < epochs; epoch++) {
                            console.log(this.train_on_batch(examples, labels));
                        }
                        _a.label = 10;
                    case 10: return [2 /*return*/];
                }
            });
        });
    };
    Model.prototype.predict = function (data) {
        if (!this.isBuilt) {
            throw "Model hasn't been build yet!..";
        }
        var exampleMatrix;
        if (data instanceof vector_1.default) {
            exampleMatrix = new matrix_1.default([data]).transpose();
        }
        else {
            exampleMatrix = [data];
        }
        this.layers[0].feedForward(exampleMatrix, false);
        for (var i = 1; i < this.layers.length; i++) {
            this.layers[i].feedForward(this.layers[i - 1], false);
        }
        return this.layers[this.layers.length - 1].activation;
    };
    Model.prototype.save = function (path) {
        var modelObj = { layers: {} };
        for (var i = 0; i < this.layers.length; i++) {
            modelObj.layers["layer_" + i] = {
                type: this.layers[i].type,
                info: this.layers[i].toSavedModel()
            };
        }
        fs.writeFileSync(path, JSON.stringify(modelObj));
    };
    Model.prototype.load = function (path) {
        var modelObj = JSON.parse(fs.readFileSync(path, { encoding: "UTF-8" }));
        var layer_keys = Object.keys(modelObj.layers).sort();
        this.layers = [];
        for (var _i = 0, layer_keys_1 = layer_keys; _i < layer_keys_1.length; _i++) {
            var layer_key = layer_keys_1[_i];
            var layer = layer_helper_1.LayerHelper.fromType(modelObj.layers[layer_key].type);
            layer.fromSavedModel(modelObj.layers[layer_key].info);
            this.layers.push(layer);
        }
        this.isBuilt = true;
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
    };
    return Model;
}());
exports.default = Model;

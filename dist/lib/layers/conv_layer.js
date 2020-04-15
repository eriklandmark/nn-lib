"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var layer_1 = __importDefault(require("./layer"));
var tensor_1 = __importDefault(require("../../tensor"));
var activations_1 = __importDefault(require("../activations/activations"));
var vector_1 = __importDefault(require("../../vector"));
var matrix_1 = __importDefault(require("../../matrix"));
var ConvolutionLayer = /** @class */ (function (_super) {
    __extends(ConvolutionLayer, _super);
    function ConvolutionLayer(nr_filters, filterSize, ch_first, activation) {
        if (nr_filters === void 0) { nr_filters = 3; }
        if (filterSize === void 0) { filterSize = [3, 3]; }
        if (ch_first === void 0) { ch_first = true; }
        var _this = _super.call(this) || this;
        _this.filterSize = [];
        _this.filters = [];
        _this.prevLayerShape = [];
        _this.padding = 0;
        _this.stride = 1;
        _this.nr_filters = 0;
        _this.errorFilters = [];
        _this.errorInput = [];
        _this.channel_first = true;
        _this.useMM = false;
        _this.channel_first = ch_first;
        _this.activationFunction = activation;
        _this.filterSize = filterSize;
        _this.nr_filters = nr_filters;
        _this.errorBias = new vector_1.default(nr_filters);
        _this.type = "conv";
        return _this;
    }
    ConvolutionLayer.prototype.buildLayer = function (prevLayerShape) {
        var h, w;
        var _a = this.filterSize, f_h = _a[0], f_w = _a[1];
        if (this.channel_first) {
            h = prevLayerShape[1];
            w = prevLayerShape[2];
        }
        else {
            h = prevLayerShape[0];
            w = prevLayerShape[1];
        }
        this.shape = [
            ((h + 2 * this.padding) - f_h + 1) / this.stride,
            ((w + 2 * this.padding) - f_w + 1) / this.stride,
            this.nr_filters
        ];
        this.prevLayerShape = prevLayerShape;
        for (var i = 0; i < this.nr_filters; i++) {
            var filter = new tensor_1.default();
            if (this.channel_first) {
                filter.createEmptyArray(prevLayerShape[0], this.filterSize[0], this.filterSize[1]);
            }
            else {
                filter.createEmptyArray(this.filterSize[0], this.filterSize[1], prevLayerShape[2]);
            }
            filter.populateRandom();
            this.filters.push(filter);
        }
        this.bias = new vector_1.default(this.nr_filters);
        this.bias.populateRandom();
    };
    ConvolutionLayer.prototype.feedForward = function (input, isInTraining, gpu) {
        if (gpu === void 0) { gpu = false; }
        if (gpu) {
        }
        else {
            var input_images = void 0;
            if (input instanceof layer_1.default) {
                input_images = input.activation;
            }
            else {
                input_images = input;
            }
            var ch = this.channel_first ? this.prevLayerShape[0] : this.prevLayerShape[2];
            var _a = this.filterSize, f_h = _a[0], f_w = _a[1];
            var patch_width_1 = this.shape[1];
            var patch_height_1 = this.shape[0];
            var new_images = [];
            if (this.useMM) {
                var filterMatrix = new matrix_1.default(this.filters.map(function (t) { return t.vectorize(true); })).transpose();
                var _loop_1 = function (t) {
                    var convolutionMatrix = filterMatrix.mm(input_images[t].im2patches(patch_height_1, patch_width_1, f_h, f_w));
                    var activationMatrix = this_1.activationFunction.normal(convolutionMatrix);
                    var convTensors = activationMatrix.rowVectors().map(function (v) { return v.reshape([patch_height_1, patch_width_1, 1]); });
                    var patch = new tensor_1.default();
                    patch.createEmptyArray(patch_height_1, patch_width_1, this_1.nr_filters);
                    var _loop_2 = function (i) {
                        patch.iterate(function (x, y, _) {
                            patch.set(x, y, i, convTensors[i].get(x, y, 0));
                        });
                    };
                    for (var i = 0; i < this_1.nr_filters; i++) {
                        _loop_2(i);
                    }
                    new_images.push(patch);
                };
                var this_1 = this;
                for (var t = 0; t < input_images.length; t++) {
                    _loop_1(t);
                }
            }
            else {
                for (var t = 0; t < input_images.length; t++) {
                    var patch = new tensor_1.default();
                    if (this.channel_first) {
                        patch.createEmptyArray(this.nr_filters, patch_height_1, patch_width_1);
                    }
                    else {
                        patch.createEmptyArray(patch_height_1, patch_width_1, this.nr_filters);
                    }
                    for (var f = 0; f < this.nr_filters; f++) {
                        for (var r = 0; r < patch_height_1; r++) {
                            for (var c = 0; c < patch_width_1; c++) {
                                var val = 0;
                                for (var c_f_c = 0; c_f_c < ch; c_f_c++) {
                                    for (var c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                        for (var c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                            if (this.channel_first) {
                                                val += input_images[t].get(c_f_c, r + c_f_h, c + c_f_w) *
                                                    this.filters[f].get(c_f_h, c_f_w, c_f_c);
                                            }
                                            else {
                                                val += input_images[t].get(r + c_f_h, c + c_f_w, c_f_c) *
                                                    this.filters[f].get(c_f_h, c_f_w, c_f_c);
                                            }
                                        }
                                    }
                                }
                                if (this.channel_first) {
                                    patch.set(f, r, c, this.activationFunction.normal(val));
                                }
                                else {
                                    patch.set(r, c, f, this.activationFunction.normal(val));
                                }
                            }
                        }
                    }
                    new_images.push(patch);
                }
            }
            this.activation = new_images;
        }
    };
    ConvolutionLayer.prototype.buildFFKernels = function (batch_size) {
        var output_shape = [this.weights.dim().c, batch_size];
        this.ff_kernel = this.gpuInstance.createKernel(function (image, filter) {
            var val = 0;
            for (var c_f_c = 0; c_f_c < this.constants.channels; c_f_c++) {
                for (var c_f_h = 0; c_f_h < this.constants.filter_height; c_f_h++) {
                    for (var c_f_w = 0; c_f_w < this.constants.filter_width; c_f_w++) {
                        if (this.constants.channel_first) {
                            val += image[c_f_c][this.thread.y + c_f_h][this.thread.x + c_f_w] * filter[c_f_h][c_f_w][c_f_c];
                        }
                        else {
                            val += image[this.thread.y + c_f_h][this.thread.x + c_f_w][c_f_c] * filter[c_f_h][c_f_w][c_f_c];
                        }
                    }
                }
            }
            return val;
        }).setConstants({
            channels: this.channel_first ? this.prevLayerShape[0] : this.prevLayerShape[2],
            filter_height: this.filterSize[0],
            filter_width: this.filterSize[1],
            channel_first: this.channel_first
        }).setPrecision("single")
            .setOutput(this.shape);
        this.ff_kernel.immutable = true;
        this.act_kernel = this.gpuInstance.createKernel(this.activationFunction.normal_gpu())
            .setPipeline(true)
            .setConstants({ softmax: this.weights.dim().c })
            .setPrecision("single")
            .setDynamicOutput(false)
            .setOutput(output_shape);
        this.act_kernel.immutable = true;
    };
    ConvolutionLayer.prototype.backPropagation = function (prev_layer, next_layer) {
        var _this = this;
        var input;
        var prev_layer_output = prev_layer.output_error;
        if (next_layer instanceof layer_1.default) {
            input = next_layer.activation;
        }
        else {
            input = next_layer;
        }
        var dout = [];
        prev_layer_output.forEach(function (t, index) {
            var ex = t.copy(false);
            ex.iterate(function (x, y, z) {
                var dActv = _this.activationFunction.derivative(_this.activation[index].get(x, y, z));
                ex.set(x, y, z, t.get(x, y, z) * dActv);
            });
            dout.push(ex);
        });
        var N = input.length;
        var _a = this.prevLayerShape, h = _a[0], w = _a[1], ch = _a[2]; // X
        var _b = this.filterSize, f_h = _b[0], f_w = _b[1]; // W
        var patch_width = dout[0].dim().c;
        var patch_height = dout[0].dim().r;
        var patch_depth = dout[0].dim().c;
        var padding_width = f_w - 1;
        var padding_height = f_h - 1;
        if (this.useMM) {
            var filterMatrix = new matrix_1.default(dout.map(function (t) { return t.vectorize(true); })).transpose();
            this.errorFilters = [];
            var _loop_3 = function (t) {
                var inputMatrix = input[t].im2patches(patch_height, patch_width, f_h, f_w);
                console.log(filterMatrix.dim(), inputMatrix.dim());
                var convolutionMatrix = inputMatrix.mm(filterMatrix);
                var convTensors = convolutionMatrix.rowVectors().map(function (v) { return v.reshape([patch_height, patch_width, 1]); });
                var patch = new tensor_1.default();
                patch.createEmptyArray(patch_height, patch_width, this_2.nr_filters);
                var _loop_5 = function (i) {
                    patch.iterate(function (x, y, _) {
                        patch.set(x, y, i, convTensors[i].get(x, y, 0));
                    });
                };
                for (var i = 0; i < this_2.nr_filters; i++) {
                    _loop_5(i);
                }
                this_2.errorFilters.push(patch);
            };
            var this_2 = this;
            for (var t = 0; t < input.length; t++) {
                _loop_3(t);
            }
            console.log(this.errorFilters);
        }
        else {
            this.errorFilters = this.filters.map(function (filter) { return filter.copy(false); });
            this.errorInput = input.map(function (inp) { return inp.copy(false); });
            this.errorBias = new vector_1.default(this.bias.size());
            for (var n = 0; n < N; n++) {
                for (var f = 0; f < this.nr_filters; f++) {
                    for (var i = 0; i < f_h; i++) {
                        for (var j = 0; j < f_w; j++) {
                            for (var k = 0; k < patch_height; k++) {
                                for (var l = 0; l < patch_width; l++) {
                                    for (var c = 0; c < ch; c++) {
                                        this.errorFilters[f].set(i, j, c, this.errorFilters[f].get(i, j, c) + (input[n].get(this.stride * i + k, this.stride * j + l, c) *
                                            dout[n].get(k, l, f)));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            var doutp_1 = new Array(dout.length).fill(new tensor_1.default());
            doutp_1.forEach(function (tensor) {
                tensor.createEmptyArray(2 * padding_height + patch_height, 2 * padding_width + patch_width, patch_depth);
            });
            for (var n = 0; n < doutp_1.length; n++) {
                for (var i = 0; i < patch_height; i++) {
                    for (var j = 0; j < patch_width; j++) {
                        for (var c = 0; c < patch_depth; c++) {
                            doutp_1[n].set(i + padding_height, j + padding_width, c, dout[n].get(i, j, c));
                        }
                    }
                }
            }
            var filterInv_1 = doutp_1.map(function (f) { return f.copy(false); });
            var _loop_4 = function (n) {
                filterInv_1[n].iterate(function (i, j, k) {
                    filterInv_1[n].set(filterInv_1[n].dim().r - 1 - i, filterInv_1[n].dim().c - 1 - j, k, doutp_1[n].get(i, j, k));
                });
            };
            for (var n = 0; n < filterInv_1.length; n++) {
                _loop_4(n);
            }
            for (var n = 0; n < N; n++) {
                for (var f = 0; f < this.nr_filters; f++) {
                    for (var i = 0; i < h + (2 * this.padding); i++) {
                        for (var j = 0; j < w + (2 * this.padding); j++) {
                            for (var k = 0; k < f_h; k++) {
                                for (var l = 0; l < f_w; l++) {
                                    for (var c = 0; c < ch; c++) {
                                        this.errorInput[n].set(i, j, c, this.errorInput[n].get(i, j, c) + (doutp_1[n].get(i + k, j + l, f) * filterInv_1[n].get(k, l, c)));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.output_error = this.errorInput;
        }
    };
    ConvolutionLayer.prototype.updateWeights = function (l_rate) {
        for (var i = 0; i < this.filters.length; i++) {
            this.filters[i] = this.filters[i].sub(this.errorFilters[i].rotate180().mul(l_rate));
        }
    };
    ConvolutionLayer.prototype.toSavedModel = function () {
        return {
            filters: this.filters.map(function (t) { return t.tensor; }),
            nr_filters: this.nr_filters,
            filterSize: this.filterSize,
            bias: this.bias.vector,
            shape: this.shape,
            activation: this.activationFunction.name,
            prevLayerShape: this.prevLayerShape
        };
    };
    ConvolutionLayer.prototype.fromSavedModel = function (data) {
        this.filters = data.filters.map(function (t) { return tensor_1.default.fromJsonObject(t); });
        this.nr_filters = data.nr_filters;
        this.filterSize = data.filterSize;
        this.bias = vector_1.default.fromJsonObj(data.bias);
        this.shape = data.shape;
        this.activationFunction = activations_1.default.fromName(data.activation);
        this.prevLayerShape = data.prevLayerShape;
    };
    return ConvolutionLayer;
}(layer_1.default));
exports.default = ConvolutionLayer;

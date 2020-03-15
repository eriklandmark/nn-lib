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
var sigmoid_1 = __importDefault(require("../activations/sigmoid"));
var ConvolutionLayer = /** @class */ (function (_super) {
    __extends(ConvolutionLayer, _super);
    function ConvolutionLayer(nr_filters, filterSize, activation) {
        if (nr_filters === void 0) { nr_filters = 3; }
        if (filterSize === void 0) { filterSize = [3, 3]; }
        if (activation === void 0) { activation = new sigmoid_1.default(); }
        var _this = _super.call(this) || this;
        _this.filterSize = [];
        _this.filters = [];
        _this.prevLayerShape = [];
        _this.padding = 0;
        _this.stride = 1;
        _this.nr_filters = 0;
        _this.errorFilters = [];
        _this.errorInput = [];
        _this.activationFunction = activation;
        _this.filterSize = filterSize;
        _this.nr_filters = nr_filters;
        _this.errorBias = new vector_1.default(nr_filters);
        _this.type = "conv";
        return _this;
    }
    ConvolutionLayer.prototype.buildLayer = function (prevLayerShape) {
        var h = prevLayerShape[0], w = prevLayerShape[1], _ = prevLayerShape[2];
        var _a = this.filterSize, f_h = _a[0], f_w = _a[1];
        this.shape = [
            ((h + 2 * this.padding) - f_h + 1) / this.stride,
            ((w + 2 * this.padding) - f_w + 1) / this.stride,
            this.nr_filters
        ];
        this.prevLayerShape = prevLayerShape;
        for (var i = 0; i < this.nr_filters; i++) {
            var filter = new tensor_1.default();
            filter.createEmptyArray(this.filterSize[0], this.filterSize[1], prevLayerShape[2]);
            filter.populateRandom();
            this.filters.push(filter);
        }
        this.bias = new vector_1.default(this.nr_filters);
        this.bias.populateRandom();
    };
    ConvolutionLayer.prototype.feedForward = function (input, isInTraining) {
        var input_images;
        if (input instanceof layer_1.default) {
            input_images = input.activation;
        }
        else {
            input_images = input;
        }
        var _a = this.prevLayerShape, h = _a[0], w = _a[1], ch = _a[2];
        var _b = this.filterSize, f_h = _b[0], f_w = _b[1];
        var patch_width = this.shape[1];
        var patch_height = this.shape[0];
        var new_images = [];
        for (var t = 0; t < input_images.length; t++) {
            var patch = new tensor_1.default();
            patch.createEmptyArray(patch_height, patch_width, this.nr_filters);
            for (var f = 0; f < this.filters.length; f++) {
                for (var r = 0; r < patch_height; r++) {
                    for (var c = 0; c < patch_width; c++) {
                        var val = 0;
                        for (var c_f_c = 0; c_f_c < ch; c_f_c++) {
                            for (var c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                for (var c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                    val += input_images[t].get(r + c_f_h, c + c_f_w, c_f_c) * this.filters[f].get(c_f_h, c_f_w, c_f_c);
                                }
                            }
                        }
                        patch.set(r, c, f, this.activationFunction.normal(val));
                    }
                }
            }
            new_images.push(patch);
        }
        this.activation = new_images;
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
        this.errorFilters = this.filters.map(function (filter) { return filter.copy(false); });
        this.errorInput = input.map(function (inp) { return inp.copy(false); });
        this.errorBias = new vector_1.default(this.bias.size());
        var dout = [];
        prev_layer_output.forEach(function (t) {
            var ex = t.copy();
            ex.iterate(function (x, y, z) {
                ex.set(x, y, z, _this.activationFunction.derivative(t.get(x, y, z)));
            });
            dout.push(ex);
        });
        var N = input.length;
        var _a = this.prevLayerShape, h = _a[0], w = _a[1], ch = _a[2]; // X
        var _b = this.filterSize, f_h = _b[0], f_w = _b[1]; // W
        var patch_width = dout[0].dim().c;
        var patch_height = dout[0].dim().r;
        var patch_depth = dout[0].dim().c;
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
        var padding_width = f_w - 1;
        var padding_height = f_h - 1;
        var doutp = new Array(dout.length).fill(new tensor_1.default());
        doutp.forEach(function (tensor) {
            tensor.createEmptyArray(2 * padding_height + patch_height, 2 * padding_width + patch_width, patch_depth);
        });
        for (var n = 0; n < doutp.length; n++) {
            for (var i = 0; i < patch_height; i++) {
                for (var j = 0; j < patch_width; j++) {
                    for (var c = 0; c < patch_depth; c++) {
                        doutp[n].set(i + padding_height, j + padding_width, c, dout[n].get(i, j, c));
                    }
                }
            }
        }
        var filterInv = doutp.map(function (f) { return f.copy(false); });
        var _loop_1 = function (n) {
            filterInv[n].iterate(function (i, j, k) {
                filterInv[n].set(filterInv[n].dim().r - 1 - i, filterInv[n].dim().c - 1 - j, k, doutp[n].get(i, j, k));
            });
        };
        for (var n = 0; n < filterInv.length; n++) {
            _loop_1(n);
        }
        for (var n = 0; n < N; n++) {
            for (var f = 0; f < this.nr_filters; f++) {
                for (var i = 0; i < h + (2 * this.padding); i++) {
                    for (var j = 0; j < w + (2 * this.padding); j++) {
                        for (var k = 0; k < f_h; k++) {
                            for (var l = 0; l < f_w; l++) {
                                for (var c = 0; c < ch; c++) {
                                    this.errorInput[n].set(i, j, c, this.errorInput[n].get(i, j, c) + (doutp[n].get(i + k, j + l, f) * filterInv[n].get(k, l, c)));
                                }
                            }
                        }
                    }
                }
            }
        }
        this.output_error = this.errorInput;
    };
    ConvolutionLayer.prototype.updateWeights = function (l_rate) {
        for (var i = 0; i < this.filters.length; i++) {
            this.filters[i] = this.filters[i].sub(this.filters[i].mul(l_rate));
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

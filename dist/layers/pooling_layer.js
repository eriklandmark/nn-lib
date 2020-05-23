"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const layer_1 = __importDefault(require("./layer"));
const tensor_1 = __importDefault(require("../tensor"));
class PoolingLayer extends layer_1.default {
    constructor(filterSize = [2, 2], stride = null, poolingFuncName = "max", ch_first = false) {
        super();
        this.type = "pooling";
        this.filterSize = [];
        this.padding = 0;
        this.stride = [];
        this.channel_first = false;
        this.channel_first = ch_first;
        this.filterSize = filterSize;
        this.stride = stride ? stride : filterSize;
        this.poolingFuncName = poolingFuncName;
    }
    buildLayer(prevLayerShape) {
        this.prevLayerShape = prevLayerShape;
        let h, w, ch;
        const [f_h, f_w] = this.filterSize;
        if (this.channel_first) {
            ch = prevLayerShape[0];
            h = prevLayerShape[1];
            w = prevLayerShape[2];
        }
        else {
            h = prevLayerShape[0];
            w = prevLayerShape[1];
            ch = prevLayerShape[2];
        }
        this.shape = [
            ((h + 2 * this.padding) - f_h) / this.stride[0] + 1,
            ((w + 2 * this.padding) - f_w) / this.stride[1] + 1,
            ch
        ];
        this.prevLayerShape = prevLayerShape;
        this.calcPoolFunc();
    }
    calcPoolFunc() {
        switch (this.poolingFuncName) {
            case "avg":
                this.poolingFunc = (x) => x.reduce((acc, n) => acc + n, 0) / x.length;
                break;
            case "max":
                this.poolingFunc = (x) => Math.max(...x);
                break;
        }
    }
    getLayerInfo() {
        const d = super.getLayerInfo();
        d.type += "_" + this.poolingFuncName;
        return d;
    }
    feedForward(input, isInTraining) {
        let input_images;
        if (input instanceof layer_1.default) {
            input_images = input.activation;
        }
        else {
            input_images = input;
        }
        let h, w, ch;
        const [f_h, f_w] = this.filterSize;
        if (this.channel_first) {
            ch = this.prevLayerShape[0];
            h = this.prevLayerShape[1];
            w = this.prevLayerShape[2];
        }
        else {
            h = this.prevLayerShape[0];
            w = this.prevLayerShape[1];
            ch = this.prevLayerShape[2];
        }
        const patch_width = this.shape[1];
        const patch_height = this.shape[0];
        let new_images;
        if (this.channel_first) {
            new_images = new tensor_1.default([input_images.shape[0], ch, patch_height, patch_width], true);
        }
        else {
            new_images = new tensor_1.default([input_images.shape[0], patch_height, patch_width, ch], true);
        }
        for (let t = 0; t < input_images.shape[0]; t++) {
            for (let f = 0; f < ch; f++) {
                for (let r = 0; r < h; r += this.stride[0]) {
                    for (let c = 0; c < w; c += this.stride[1]) {
                        let val = [];
                        for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                            for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                if (this.channel_first) {
                                    val.push(input_images.t[t][f][r + c_f_h][c + c_f_w]);
                                }
                                else {
                                    val.push(input_images.t[t][r + c_f_h][c + c_f_w][f]);
                                }
                            }
                        }
                        if (this.channel_first) {
                            new_images.t[t][f][r / this.stride[0]][c / this.stride[1]] = this.poolingFunc(val);
                        }
                        else {
                            new_images.t[t][r / this.stride[0]][c / this.stride[1]][f] = this.poolingFunc(val);
                        }
                    }
                }
            }
        }
        this.activation = new_images;
    }
    backPropagation(prev_layer, next_layer) {
        let input;
        if (next_layer instanceof layer_1.default) {
            input = next_layer.activation;
        }
        else {
            input = next_layer;
        }
        let t = new tensor_1.default([prev_layer.output_error.shape[0], this.prevLayerShape[0], this.prevLayerShape[1], this.prevLayerShape[2]], true);
        const [s_h, s_w] = this.stride;
        const [h, w, d] = this.prevLayerShape;
        const [hh, ww] = this.shape;
        const [f_h, f_w] = this.filterSize;
        for (let n = 0; n < t.shape[0]; n++) {
            for (let ch = 0; ch < d; ch++) {
                for (let r = 0; r < hh; r++) {
                    for (let c = 0; c < ww; c++) {
                        let i = -1;
                        let j = -1;
                        for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                            for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                if (input.t[n][(r * s_h) + c_f_h][(c * s_w) + c_f_w][ch] == this.activation.t[n][r][c][ch]) {
                                    i = c_f_h;
                                    j = c_f_w;
                                    break;
                                }
                            }
                        }
                        t.t[n][(r * s_h) + i][(c * s_w) + j][ch] = prev_layer.output_error.t[n][r][c][ch];
                    }
                }
            }
        }
        this.output_error = t;
    }
    updateLayer() { }
    toSavedModel() {
        const data = super.toSavedModel();
        data.layer_specific = {
            filterSize: this.filterSize,
            poolingFuncName: this.poolingFuncName,
            padding: this.padding,
            stride: this.stride
        };
        return data;
    }
    fromSavedModel(data) {
        super.fromSavedModel(data);
        this.filterSize = data.layer_specific.filterSize;
        this.poolingFuncName = data.layer_specific.poolingFuncName;
        this.stride = data.layer_specific.stride;
        this.padding = data.layer_specific.padding;
        this.calcPoolFunc();
    }
}
exports.default = PoolingLayer;

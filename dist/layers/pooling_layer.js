"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const layer_1 = __importDefault(require("./layer"));
const tensor_1 = __importDefault(require("../tensor"));
class PoolingLayer extends layer_1.default {
    constructor(filterSize = [2, 2], stride = null, ch_first = false) {
        super();
        this.type = "pooling";
        this.prevShape = [];
        this.filterSize = [];
        this.padding = 0;
        this.stride = [];
        this.channel_first = true;
        this.poolingFunc = "max";
        this.channel_first = ch_first;
        this.filterSize = filterSize;
        this.stride = stride ? stride : filterSize;
    }
    buildLayer(prevLayerShape) {
        this.prevShape = prevLayerShape;
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
        console.log(h, (h + 2 * this.padding) - f_h / this.stride[0]);
        this.prevLayerShape = prevLayerShape;
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
        let new_images = [];
        for (let t = 0; t < input_images.length; t++) {
            let patch = new tensor_1.default();
            if (this.channel_first) {
                patch.createEmptyArray(ch, patch_height, patch_width);
            }
            else {
                patch.createEmptyArray(patch_height, patch_width, ch);
            }
            for (let f = 0; f < ch; f++) {
                for (let r = 0; r < h; r += this.stride[0]) {
                    for (let c = 0; c < w; c += this.stride[1]) {
                        let val = [];
                        for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                            for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                if (this.channel_first) {
                                    val.push(input_images[t].get(f, r + c_f_h, c + c_f_w));
                                }
                                else {
                                    val.push(input_images[t].get(r + c_f_h, c + c_f_w, f));
                                }
                            }
                        }
                        if (this.channel_first) {
                            patch.set(f, r / this.stride[0], c / this.stride[1], Math.max(...val));
                        }
                        else {
                            patch.set(r / this.stride[0], c / this.stride[1], f, Math.max(...val));
                        }
                    }
                }
            }
            new_images.push(patch);
        }
        this.activation = new_images;
    }
    backPropagation(prev_layer, next_layer) {
        const gradients = prev_layer.output_error;
        let input;
        if (next_layer instanceof layer_1.default) {
            input = next_layer.activation;
        }
        else {
            input = next_layer;
        }
        let t = new Array(gradients.length);
        for (let i = 0; i < t.length; i++) {
            t[i] = new tensor_1.default();
            t[i].createEmptyArray(this.prevShape[0], this.prevShape[1], this.prevShape[2]);
        }
        const [s_h, s_w] = this.stride;
        const [h, w, d] = this.prevShape;
        const [hh, ww] = this.shape;
        const [f_h, f_w] = this.filterSize;
        for (let n = 0; n < t.length; n++) {
            for (let ch = 0; ch < d; ch++) {
                for (let r = 0; r < hh; r++) {
                    for (let c = 0; c < ww; c++) {
                        let i = -1;
                        let j = -1;
                        for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                            for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                if (input[n].get((r * s_h) + c_f_h, (c * s_w) + c_f_w, ch) == this.activation[n].get(r, c, ch)) {
                                    i = c_f_h;
                                    j = c_f_w;
                                    break;
                                }
                            }
                        }
                        t[n].set((r * s_h) + i, (c * s_w) + j, ch, gradients[n].get(r, c, ch));
                    }
                }
            }
        }
        this.output_error = t;
    }
    toSavedModel() {
        return {
            filterSize: this.filterSize,
            shape: this.shape,
            prevLayerShape: this.prevLayerShape,
            poolingFunc: this.poolingFunc,
            padding: this.padding,
            stride: this.stride
        };
    }
    fromSavedModel(data) {
        this.filterSize = data.filterSize;
        this.shape = data.shape;
        this.prevLayerShape = data.prevLayerShape;
        this.poolingFunc = data.poolingFunc;
        this.stride = data.stride;
        this.padding = data.padding;
    }
}
exports.default = PoolingLayer;

import Layer from "./layer";
import Tensor from "../tensor";
import {SavedLayer} from "../model";

export default class PoolingLayer extends Layer {

    type: string = "pooling"
    filterSize: number[] = []
    padding: number = 0;
    stride: number[] = [];
    channel_first:boolean = true
    poolingFunc: string = "max"

    constructor(filterSize: number[] = [2, 2], stride: number[] = null, ch_first: boolean = false) {
        super();
        this.channel_first = ch_first
        this.filterSize = filterSize
        this.stride = stride? stride : filterSize
    }

    buildLayer(prevLayerShape: number[]) {
        this.prevLayerShape = prevLayerShape
        let h, w, ch;
        const [f_h, f_w] = this.filterSize
        if (this.channel_first) {
            ch = prevLayerShape[0]
            h = prevLayerShape[1]
            w = prevLayerShape[2]
        } else {
            h = prevLayerShape[0]
            w = prevLayerShape[1]
            ch = prevLayerShape[2]
        }

        this.shape = [
            ((h + 2 * this.padding) - f_h )/ this.stride[0] + 1,
            ((w + 2 * this.padding) - f_w )/ this.stride[1] + 1,
            ch
        ]
        console.log(h, (h + 2 * this.padding) - f_h / this.stride[0])
        this.prevLayerShape = prevLayerShape
    }

    getLayerInfo(): { shape: number[]; type: string; activation: string } {
        const d =  super.getLayerInfo();
        d.type += "_" + this.poolingFunc
        return d
    }

    feedForward(input: Layer, isInTraining: boolean) {
        let input_images: Tensor[]
        if (input instanceof Layer) {
            input_images = <Tensor[]>input.activation
        } else {
            input_images = <Tensor[]>input
        }
        let h, w, ch;
        const [f_h, f_w] = this.filterSize
        if (this.channel_first) {
            ch = this.prevLayerShape[0]
            h = this.prevLayerShape[1]
            w = this.prevLayerShape[2]
        } else {
            h = this.prevLayerShape[0]
            w = this.prevLayerShape[1]
            ch = this.prevLayerShape[2]
        }
        const patch_width = this.shape[1]
        const patch_height = this.shape[0]
        let new_images: Tensor[] = []
        for (let t = 0; t < input_images.length; t++) {
            let patch = new Tensor();
            if (this.channel_first) {
                patch.createEmptyArray(ch, patch_height, patch_width)
            } else {
                patch.createEmptyArray(patch_height, patch_width, ch)
            }

            for (let f = 0; f < ch; f++) {
                for (let r = 0; r < h; r += this.stride[0]) {
                    for (let c = 0; c < w; c += this.stride[1]) {
                        let val: number[] = []
                        for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                            for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                if (this.channel_first) {
                                    val.push(input_images[t].get(f, r + c_f_h, c + c_f_w))
                                } else {
                                    val.push(input_images[t].get(r + c_f_h, c + c_f_w, f))
                                }
                            }
                        }
                        if(this.channel_first) {
                            patch.set(f, r/this.stride[0], c/this.stride[1], Math.max(...val))
                        } else {
                            patch.set(r/this.stride[0], c/this.stride[1], f, Math.max(...val))
                        }
                    }
                }
            }
            new_images.push(patch)
        }

        this.activation = new_images
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]) {
        const gradients = <Tensor[]> prev_layer.output_error
        let input: Tensor[]
        if (next_layer instanceof Layer) {
            input = <Tensor[]> next_layer.activation
        } else {
            input = next_layer
        }

        let t: Tensor[] = new Array(gradients.length);
        for(let i = 0; i < t.length; i++) {
            t[i] = new Tensor();
            t[i].createEmptyArray(this.prevLayerShape[0], this.prevLayerShape[1], this.prevLayerShape[2])
        }

        const [s_h,s_w] = this.stride
        const [h, w, d] = this.prevLayerShape
        const [hh, ww] = this.shape
        const [f_h, f_w] = this.filterSize
        for(let n = 0; n < t.length; n++) {
            for (let ch = 0; ch < d; ch++) {
                for (let r = 0; r < hh; r++) {
                    for (let c = 0; c < ww; c++) {
                        let i = -1
                        let j = -1
                        for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                            for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                if(input[n].get((r*s_h) + c_f_h, (c*s_w) + c_f_w, ch) == this.activation[n].get(r,c,ch)) {
                                    i = c_f_h
                                    j = c_f_w
                                    break
                                }
                            }
                        }
                        t[n].set((r*s_h) + i,(c*s_w) + j, ch, gradients[n].get(r, c, ch))
                    }
                }
            }
        }

        this.output_error = t
    }

    toSavedModel(): SavedLayer {
        const data = super.toSavedModel()
        data.layer_specific = {
            filterSize: this.filterSize,
            poolingFunc: this.poolingFunc,
            padding: this.padding,
            stride: this.stride
        }

        return data
    }

    fromSavedModel(data: SavedLayer) {
        super.fromSavedModel(data)
        this.filterSize = data.layer_specific.filterSize
        this.poolingFunc = data.layer_specific.poolingFunc
        this.stride = <number[]> data.layer_specific.stride
        this.padding = data.layer_specific.padding
    }

}
import Layer from "./layer";
import Tensor from "../../tensor";
import IActivation from "../activations/activations";
import Vector from "../../vector";

export default class ConvolutionLayer extends Layer {

    filterSize: number[] = []
    filters: Tensor[] = []
    prevLayerShape: number[] = []
    padding: number = 0;
    stride: number = 1;
    nr_filters: number = 0
    errorFilters: Tensor[] = []
    errorInput: Tensor[] = []

    constructor(nr_filters: number = 3, filterSize: number[], activation: IActivation) {
        super(activation);
        this.filterSize = filterSize
        this.nr_filters = nr_filters
        this.errorBias = new Vector(nr_filters)
    }

    buildLayer(prevLayerShape: number[]) {
        const [h, w, _] = prevLayerShape
        const [f_h, f_w] = this.filterSize
        this.shape = [
            ((h + 2 * this.padding) - f_h + 1) / this.stride,
            ((w + 2 * this.padding) - f_w + 1) / this.stride,
            this.nr_filters
        ]
        this.prevLayerShape = prevLayerShape
        for (let i = 0; i < this.nr_filters; i++) {
            const filter = new Tensor()
            filter.createEmptyArray(this.filterSize[0], this.filterSize[1], prevLayerShape[2])
            filter.populateRandom()
            this.filters.push(filter)
        }
        this.bias = new Vector(this.nr_filters)
        this.bias.populateRandom()
    }

    feedForward(input: Layer | Tensor[], isInTraining: boolean) {
        let input_images: Tensor[]
        if (input instanceof Layer) {
            input_images = <Tensor[]>input.activation
        } else {
            input_images = <Tensor[]>input
        }

        const [h, w, ch] = this.prevLayerShape
        const [f_h, f_w] = this.filterSize
        const patch_width = ((w + 2 * this.padding) - f_w + 1) / this.stride
        const patch_height = ((h + 2 * this.padding) - f_h + 1) / this.stride
        console.log(h,w,ch)
        console.log(f_h,f_w)
        console.log(patch_width, patch_height)

        let new_images: Tensor[] = []

        for (let t = 0; t < input_images.length; t++) {
            let patch = new Tensor();
            patch.createEmptyArray(patch_height, patch_width, this.nr_filters)
            for (let f = 0; f < this.filters.length; f++) {
                for (let r = 0; r < patch_height; r++) {
                    for (let c = 0; c < patch_width; c++) {
                        let val: number = 0
                        for (let c_f_c = 0; c_f_c < ch; c_f_c++) {
                            for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                    val += input_images[t].get(r + c_f_h, c + c_f_w, c_f_c) * this.filters[f].get(c_f_h, c_f_w, c_f_c)
                                }
                            }
                        }
                        patch.set(r, c, f, val)
                    }
                }
            }
            new_images.push(patch)
        }

        this.activation = new_images
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]) {
        let input: Tensor[]
        let dout: Tensor[] = <Tensor[]> prev_layer.output_error
        if (next_layer instanceof Layer) {
            input = <Tensor[]> next_layer.activation
        } else {
            input = next_layer
        }
        this.errorFilters = this.filters.map((filter) => filter.copy(false))
        this.errorInput = input.map((inp) => inp.copy(false))
        this.errorBias = new Vector(this.bias.size())

        const N = input.length
        const [h, w, ch] = this.prevLayerShape // X
        const [f_h, f_w] = this.filterSize // W
        const patch_width = dout[0].dim().c
        const patch_height =  dout[0].dim().r
        const patch_depth =  dout[0].dim().c

        for (let n = 0; n < N; n++) {
            for (let f = 0; f < this.nr_filters; f++) {
                for (let i = 0; i < f_h; i++) {
                    for (let j = 0; j < f_w; j++) {
                        for (let k = 0; k < patch_height; k++) {
                            for (let l = 0; l < patch_width; l++) {
                                for (let c = 0; c < ch; c++) {
                                    this.errorFilters[f].set(i,j,c,
                                        this.errorFilters[f].get(i,j,c) + (
                                            input[n].get(this.stride*i+k, this.stride*j+l, c) *
                                            dout[n].get(k, l, f)
                                        ))
                                }
                            }
                        }
                    }
                }
            }
        }

        const padding_width = f_w - 1
        const padding_height = f_h - 1
        const doutp: Tensor[] = new Array(dout.length).fill(new Tensor())
        doutp.forEach((tensor) => {
            tensor.createEmptyArray(2 * padding_height + patch_height, 2 * padding_width + patch_width, patch_depth)
        })

        for (let n = 0; n < doutp.length; n++) {
            for (let i = 0; i < patch_height; i++) {
                for (let j = 0; j < patch_width; j++) {
                    for (let c = 0; c < patch_depth; c++) {
                        doutp[n].set(i + padding_height, j + padding_width, c, dout[n].get(i,j,c))
                    }
                }
            }
        }

        const filterInv = doutp.map((f) => f.copy(false))
        for (let n = 0; n < filterInv.length; n++) {
            filterInv[n].iterate((i: number, j: number, k: number) => {
                filterInv[n].set(filterInv[n].dim().r - 1 - i, filterInv[n].dim().c - 1 - j , k, doutp[n].get(i,j,k))
            })
        }


        for (let n = 0; n < N; n++) {
            for (let f = 0; f < this.nr_filters; f++) {
                for (let i = 0; i < h + (2 * this.padding); i++) {
                    for (let j = 0; j < w + (2 * this.padding); j++) {
                        for (let k = 0; k < f_h; k++) {
                            for (let l = 0; l < f_w; l++) {
                                for (let c = 0; c < ch; c++) {
                                    this.errorInput[n].set(i,j,c,
                                        this.errorInput[n].get(i,j,c) + (
                                            doutp[n].get(i+k, j+l, f) * filterInv[f].get(k, l, c)
                                        ))
                                }
                            }
                        }
                    }
                }
            }
        }



    }

    updateWeights(l_rate: number) {}
}
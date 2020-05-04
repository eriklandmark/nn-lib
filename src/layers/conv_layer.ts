import Layer from "./layer";
import Tensor from "../tensor";
import {IActivation} from "../activations/activations";
import Vector from "../vector";
import {SavedLayer} from "../model";
import Matrix from "../matrix";

export default class ConvolutionLayer extends Layer {

    weights: Tensor[] = []
    filterSize: number[] = []
    padding: number = 0;
    stride: number = 1;
    nr_filters: number = 0
    errorWeights: Tensor[] = []
    channel_first:boolean = true

    ff_kernel: any
    act_kernel: any
    bp_error_kernel: any
    bp_error_weight_kernel: any

    useMM: boolean = false
    use_bias: boolean = true

    constructor(nr_filters: number = 3, filterSize: number[] = [3, 3], ch_first: boolean = false,
                activation: IActivation, use_bias: boolean = true) {
        super();
        this.channel_first = ch_first
        this.activationFunction = activation
        this.filterSize = filterSize
        this.nr_filters = nr_filters

        this.type = "conv"
        this.use_bias = use_bias
    }

    buildLayer(prevLayerShape: number[]) {
        let h, w;
        const [f_h, f_w] = this.filterSize
        if (this.channel_first) {
            h = prevLayerShape[1]
            w = prevLayerShape[2]
        } else {
            h = prevLayerShape[0]
            w = prevLayerShape[1]
        }

        this.shape = [
            ((h + 2 * this.padding) - f_h + 1) / this.stride,
            ((w + 2 * this.padding) - f_w + 1) / this.stride,
            this.nr_filters
        ]
        this.prevLayerShape = prevLayerShape

        for (let i = 0; i < this.nr_filters; i++) {
            const filter = new Tensor()
            if (this.channel_first) {
                filter.createEmptyArray(prevLayerShape[0], this.filterSize[0], this.filterSize[1])
            } else {
                filter.createEmptyArray(this.filterSize[0], this.filterSize[1], prevLayerShape[2])
            }

            filter.populateRandom()
            this.weights.push(filter)
        }
        this.bias = new Matrix()
        this.bias.createEmptyArray(1, this.nr_filters)
        this.bias.populateRandom()
        this.errorBias = new Matrix()
        this.errorBias.createEmptyArray(1, this.nr_filters)
    }

    feedForward(input: Layer | Tensor[], isInTraining: boolean) {
        if (false) {

        } else {
            let input_images: Tensor[]
            if (input instanceof Layer) {
                input_images = <Tensor[]>input.activation
            } else {
                input_images = <Tensor[]>input
            }
            const ch = this.channel_first? this.prevLayerShape[0]: this.prevLayerShape[2];
            const [f_h, f_w] = this.filterSize
            const patch_width = this.shape[1]
            const patch_height = this.shape[0]
            let new_images: Tensor[] = []
            if (this.useMM) {
                const filterMatrix = new Matrix(this.weights.map((t) => t.vectorize(true))).transpose()
                for (let t = 0; t < input_images.length; t++) {
                    const convolutionMatrix = <Matrix> filterMatrix.mm(input_images[t].im2patches(patch_height, patch_width, f_h, f_w))
                    const activationMatrix = <Matrix> this.activationFunction.normal(convolutionMatrix)
                    const convTensors = activationMatrix.rowVectors().map((v) => v.reshape([patch_height, patch_width, 1]))
                    const patch: Tensor = new Tensor()
                    patch.createEmptyArray(patch_height, patch_width, this.nr_filters)
                    for (let i = 0; i < this.nr_filters; i++) {
                        patch.iterate((x, y, _) => {
                            patch.set(x, y, i, convTensors[i].get(x,y,0))
                        })
                    }
                    new_images.push(patch)
                }
            }else{
                for (let t = 0; t < input_images.length; t++) {
                    let patch = new Tensor();
                    if (this.channel_first) {
                        patch.createEmptyArray(this.nr_filters, patch_height, patch_width)
                    } else {
                        patch.createEmptyArray(patch_height, patch_width, this.nr_filters)
                    }

                    for (let f = 0; f < this.nr_filters; f++) {
                        for (let r = 0; r < patch_height; r++) {
                            for (let c = 0; c < patch_width; c++) {
                                let val: number = 0
                                for (let c_f_c = 0; c_f_c < ch; c_f_c++) {
                                    for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                        for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                            if (this.channel_first) {
                                                val += input_images[t].get(c_f_c, r + c_f_h, c + c_f_w) *
                                                    this.weights[f].get(c_f_h, c_f_w, c_f_c)
                                            } else {
                                                //console.log(r + c_f_h, c + c_f_w, c_f_c)
                                                val += input_images[t].get(r + c_f_h, c + c_f_w, c_f_c) *
                                                    this.weights[f].get(c_f_h, c_f_w, c_f_c)
                                            }
                                        }
                                    }
                                }
                                if(this.channel_first) {
                                    patch.set(f, r, c, this.activationFunction.normal(val) + (this.use_bias? this.bias.get(0,f) : 0))
                                } else {
                                    patch.set(r, c, f, this.activationFunction.normal(val) + (this.use_bias? this.bias.get(0,f) : 0))
                                }
                            }
                        }
                    }
                    new_images.push(patch)
                }
            }
            this.activation = new_images
        }
    }

    /*buildFFKernels(batch_size: number) {
        const output_shape = [this.weights.dim().c, batch_size]
        this.ff_kernel = this.gpuInstance.createKernel(function (image, filter) {
            let val: number = 0
            for (let c_f_c = 0; c_f_c < this.constants.channels; c_f_c++) {
                for (let c_f_h = 0; c_f_h < this.constants.filter_height; c_f_h++) {
                    for (let c_f_w = 0; c_f_w < this.constants.filter_width; c_f_w++) {
                        if (this.constants.channel_first) {
                            val += image[c_f_c][this.thread.y + c_f_h][this.thread.x + c_f_w] * filter[c_f_h][c_f_w][c_f_c]
                        } else {
                            val += image[this.thread.y + c_f_h][this.thread.x + c_f_w][c_f_c] * filter[c_f_h][c_f_w][c_f_c]
                        }
                    }
                }
            }
            return val;
        }).setConstants({
            channels: this.channel_first? this.prevLayerShape[0]: this.prevLayerShape[2],
            filter_height: this.filterSize[0],
            filter_width: this.filterSize[1],
            channel_first: this.channel_first
        }).setPrecision("single")
            .setOutput(this.shape)
        this.ff_kernel.immutable = true

        this.act_kernel = this.gpuInstance.createKernel(this.activationFunction.normal_gpu())
            .setPipeline(true)
            .setConstants({softmax: this.weights.dim().c})
            .setPrecision("single")
            .setDynamicOutput(false)
            .setOutput(output_shape)
        this.act_kernel.immutable = true
    }*/

    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]) {
        let input: Tensor[]
        let prev_layer_output: Tensor[] = <Tensor[]> prev_layer.output_error
        if (next_layer instanceof Layer) {
            input = <Tensor[]> next_layer.activation
        } else {
            input = next_layer
        }

        let dout: Tensor[] = []
        prev_layer_output.forEach((t, index )=> {
            let ex = t.copy(false);
            ex.iterate((x: number, y : number, z: number) => {
                const dActv = <number> this.activationFunction.derivative((<Tensor> this.activation[index]).get(x,y,z))
                ex.set(x,y,z, t.get(x,y,z) * dActv)
            })
            dout.push(ex.rotate180())
        })

        const N = input.length
        const [h, w, ch] = this.prevLayerShape // X
        const [f_h, f_w] = this.filterSize // W
        const patch_width = dout[0].dim().c
        const patch_height =  dout[0].dim().r
        const patch_depth =  dout[0].dim().d
        const padding_width = f_w - 1
        const padding_height = f_h - 1

        if (this.useMM) {
            const filterMatrix = new Matrix(dout.map((t) => t.vectorize(true))).transpose()
            this.errorWeights = []

            for (let t = 0; t < input.length; t++) {
                const inputMatrix = <Matrix> input[t].im2patches(patch_height, patch_width, f_h, f_w)
                console.log(filterMatrix.dim(), inputMatrix.dim())
                const convolutionMatrix = <Matrix> inputMatrix.mm(filterMatrix)
                const convTensors = convolutionMatrix.rowVectors().map((v) => v.reshape([patch_height, patch_width, 1]))
                const patch: Tensor = new Tensor()
                patch.createEmptyArray(patch_height, patch_width, this.nr_filters)
                for (let i = 0; i < this.nr_filters; i++) {
                    patch.iterate((x, y, _) => {
                        patch.set(x, y, i, convTensors[i].get(x,y,0))
                    })
                }
                this.errorWeights.push(patch)
            }

        } else {
            this.errorWeights = this.weights.map((filter) => filter.copy(false))
            this.output_error = input.map((inp) => inp.copy(false))

            for (let n = 0; n < N; n++) {
                for (let f = 0; f < this.nr_filters; f++) {
                    for (let i = 0; i < f_h; i++) {
                        for (let j = 0; j < f_w; j++) {
                            for (let k = 0; k < patch_height; k++) {
                                for (let l = 0; l < patch_width; l++) {
                                    for (let c = 0; c < ch; c++) {
                                        this.errorWeights[f].set(i, j, c,
                                            this.errorWeights[f].get(i, j, c) + (
                                                input[n].get(this.stride * i + k, this.stride * j + l, c) *
                                                dout[n].get(k, l, f)
                                            ))
                                    }
                                }
                            }
                        }
                    }
                }
            }

            this.errorWeights = this.errorWeights.map((eW) => eW.rotate180())

            if(this.use_bias) {
                const sum: Matrix[] = []

                for(let n = 0; n < dout.length; n++) {
                    const sumMatrix = new Matrix()
                    sumMatrix.createEmptyArray(1, dout[n].dim().d)
                    dout[n].iterate((i,j,k) => {
                        sumMatrix.set(0,k, sumMatrix.get(0, k) + dout[n].get(i,j,k))
                    })
                    sum.push(sumMatrix)
                }


                this.errorBias = sum.reduce((acc, v) => acc.add(v),
                    sum[0].copy(false)).div(sum.length)
            }


            if (!this.isFirstLayer) {
                const doutp: Tensor[] = new Array(dout.length).fill(new Tensor())
                doutp.forEach((tensor) => {
                    tensor.createEmptyArray(2 * padding_height + patch_height, 2 * padding_width + patch_width, patch_depth)
                })

                for (let n = 0; n < doutp.length; n++) {
                    for (let i = 0; i < patch_height; i++) {
                        for (let j = 0; j < patch_width; j++) {
                            for (let c = 0; c < patch_depth; c++) {
                                doutp[n].set(i + padding_height, j + padding_width, c, dout[n].get(i, j, c))
                            }
                        }
                    }
                }

                const filterInv = this.weights.map((f) => f.copy(false))
                for (let n = 0; n < filterInv.length; n++) {
                    filterInv[n].iterate((i: number, j: number, k: number) => {
                        filterInv[n].set(filterInv[n].dim().r - 1 - i, filterInv[n].dim().c - 1 - j, k, this.weights[n].get(i, j, k))
                    })
                }

                for (let n = 0; n < N; n++) {
                    for (let f = 0; f < this.nr_filters; f++) {
                        for (let i = 0; i < h + (2 * this.padding); i++) {
                            for (let j = 0; j < w + (2 * this.padding); j++) {
                                for (let k = 0; k < f_h; k++) {
                                    for (let l = 0; l < f_w; l++) {
                                        for (let c = 0; c < ch; c++) {
                                            this.output_error[n].set(i, j, c,
                                                this.output_error[n].get(i, j, c) + (
                                                    doutp[n].get(i + k, j + l, f) * filterInv[f].get(k, l, c)
                                                ))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    convolve(image: Tensor, filters: Tensor[], channel_first= false) {
        const f_h = filters[0].dim().r
        const f_w = filters[0].dim().c
        const patch_width = ((image.dim().r + 2 * this.padding) - f_h + 1) / this.stride
        const patch_height = ((image.dim().c + 2 * this.padding) - f_w + 1) / this.stride
        if (this.useMM) {
            const filterMatrix = new Matrix(filters.map((t) => t.vectorize(true))).transpose()

            return (<Matrix>filterMatrix.mm(image.im2patches(patch_height, patch_width, filters[0].dim().r, filters[0].dim().c)))
                .rowVectors().map((v) => v.reshape([patch_height, patch_width, 1]))
        } else {
            let patch = new Tensor();
            if (channel_first) {
                patch.createEmptyArray(filters.length, patch_height, patch_width)
            } else {
                patch.createEmptyArray(patch_height, patch_width, filters.length)
            }

            const chs = channel_first ? image.dim().r : image.dim().d
            for (let f = 0; f < filters.length; f++) {
                for (let r = 0; r < patch_height; r++) {
                    for (let c = 0; c < patch_width; c++) {
                        let val: number = 0
                        for (let c_f_c = 0; c_f_c < chs; c_f_c++) {
                            for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                    if (channel_first) {
                                        val += image.get(c_f_c, r + c_f_h, c + c_f_w) * filters[f].get(c_f_h, c_f_w, c_f_c)
                                    } else {
                                        val += image.get(r + c_f_h, c + c_f_w, c_f_c) * filters[f].get(c_f_h, c_f_w, c_f_c)
                                    }
                                }
                            }
                        }
                        if (channel_first) {
                            patch.set(f, r, c, val)
                        } else {
                            patch.set(r, c, f, val)
                        }
                    }
                }
            }
            return patch
        }
    }

    toSavedModel(): SavedLayer {
        const data = super.toSavedModel()
        data.layer_specific = {
            nr_filters: this.nr_filters,
            filterSize: this.filterSize,
            use_bias: this.use_bias
        }

        return data
    }

    updateLayer() {
        this.optimizer.optimizeWeights()
        if (this.use_bias) {
            this.optimizer.optimizeBias()
        }
    }

    fromSavedModel(data: SavedLayer) {
        super.fromSavedModel(data)
        this.nr_filters = data.layer_specific.nr_filters
        this.filterSize = data.layer_specific.filterSize
        this.use_bias = data.layer_specific.use_bias
    }
}
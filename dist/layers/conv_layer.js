import Layer from "./layer";
import Tensor from "../tensor";
export default class ConvolutionLayer extends Layer {
    constructor(nr_filters = 3, filterSize = [3, 3], ch_first = false, activation, use_bias = true) {
        super();
        this.weights = new Tensor();
        this.filterSize = [];
        this.padding = 0;
        this.stride = 1;
        this.nr_filters = 0;
        this.errorWeights = new Tensor();
        this.channel_first = true;
        this.useMM = false;
        this.use_bias = true;
        this.channel_first = ch_first;
        this.activationFunction = activation;
        this.filterSize = filterSize;
        this.nr_filters = nr_filters;
        this.type = "conv";
        this.use_bias = use_bias;
    }
    buildLayer(prevLayerShape) {
        let h, w;
        const [f_h, f_w] = this.filterSize;
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
        if (this.channel_first) {
            this.weights = new Tensor([this.nr_filters, prevLayerShape[0], this.filterSize[0], this.filterSize[1]], true);
        }
        else {
            this.weights = new Tensor([this.nr_filters, this.filterSize[0], this.filterSize[1], prevLayerShape[2]], true);
        }
        this.weights.populateRandom();
        this.bias = new Tensor([1, this.nr_filters], true);
        this.bias.populateRandom();
        this.errorBias = new Tensor([1, this.nr_filters], true);
    }
    feedForward(input, isInTraining) {
        if (false) {
        }
        else {
            let input_images;
            if (input instanceof Layer) {
                input_images = input.activation;
            }
            else {
                input_images = input;
            }
            const ch = this.channel_first ? this.prevLayerShape[0] : this.prevLayerShape[2];
            const [f_h, f_w] = this.filterSize;
            const patch_width = this.shape[1];
            const patch_height = this.shape[0];
            let new_images;
            if (this.useMM) {
                /*const filterMatrix = new Tensor(this.weights.map((t) => t.vectorize(true))).transpose()
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
                }*/
            }
            else {
                if (this.channel_first) {
                    new_images = new Tensor([input_images.shape[0], this.nr_filters, patch_height, patch_width], true);
                }
                else {
                    new_images = new Tensor([input_images.shape[0], patch_height, patch_width, this.nr_filters], true);
                }
                for (let n = 0; n < input_images.shape[0]; n++) {
                    for (let f = 0; f < this.nr_filters; f++) {
                        for (let r = 0; r < patch_height; r++) {
                            for (let c = 0; c < patch_width; c++) {
                                let val = 0;
                                for (let c_f_c = 0; c_f_c < ch; c_f_c++) {
                                    for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                        for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                            if (this.channel_first) {
                                                val += input_images.t[n][c_f_c][r + c_f_h][c + c_f_w] *
                                                    this.weights.t[f][c_f_h][c_f_w][c_f_c];
                                            }
                                            else {
                                                val += input_images.t[n][r + c_f_h][c + c_f_w][c_f_c] *
                                                    this.weights.t[f][c_f_h][c_f_w][c_f_c];
                                            }
                                        }
                                    }
                                }
                                if (this.channel_first) {
                                    new_images.t[n][f][r][c] = this.activationFunction.normal(val) + (this.use_bias ? this.bias.t[0][f] : 0);
                                }
                                else {
                                    new_images.t[n][r][c][f] = this.activationFunction.normal(val) + (this.use_bias ? this.bias.t[0][f] : 0);
                                }
                            }
                        }
                    }
                }
            }
            this.activation = new_images;
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
    backPropagation(prev_layer, next_layer) {
        let input;
        if (next_layer instanceof Layer) {
            input = next_layer.activation;
        }
        else {
            input = next_layer;
        }
        const dout = prev_layer.output_error.mul(this.activationFunction.derivative(this.activation)).rotate180();
        const N = input.shape[0];
        const [h, w, ch] = this.prevLayerShape; // X
        const [f_h, f_w] = this.filterSize; // W
        const patch_width = dout.shape[2];
        const patch_height = dout.shape[1];
        const patch_depth = dout.shape[3];
        const padding_width = f_w - 1;
        const padding_height = f_h - 1;
        if (this.useMM) {
            /*const filterMatrix = new Tensor(dout.map((t) => t.vectorize(true))).transpose()
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
            */
        }
        else {
            this.errorWeights = this.weights.copy();
            this.output_error = input.copy();
            for (let n = 0; n < N; n++) {
                for (let f = 0; f < this.nr_filters; f++) {
                    for (let i = 0; i < f_h; i++) {
                        for (let j = 0; j < f_w; j++) {
                            for (let k = 0; k < patch_height; k++) {
                                for (let l = 0; l < patch_width; l++) {
                                    for (let c = 0; c < ch; c++) {
                                        this.errorWeights.t[f][i][j][c] =
                                            this.errorWeights.t[f][i][j][c] + (input.t[n][this.stride * i + k][this.stride * j + l][c] *
                                                dout.t[n][k][l][f]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.errorWeights = this.errorWeights.rotate180();
            if (this.use_bias) {
                const sum = new Tensor([1, this.nr_filters], true);
                dout.iterate((n, i, j, k) => { sum.t[0][k] = sum.t[0][k] + dout.t[n][i][j][k]; });
                this.errorBias = sum.div(dout.shape[0]);
            }
            if (!this.isFirstLayer) {
                const doutp = new Tensor([dout.shape[0], 2 * padding_height + patch_height, 2 * padding_width + patch_width, patch_depth], true);
                for (let n = 0; n < doutp.shape[0]; n++) {
                    for (let i = 0; i < patch_height; i++) {
                        for (let j = 0; j < patch_width; j++) {
                            for (let c = 0; c < patch_depth; c++) {
                                doutp.t[n][i + padding_height][j + padding_width][c] = dout.t[n][i][j][c];
                            }
                        }
                    }
                }
                const filterInv = this.weights.copy(false);
                filterInv.iterate((n, i, j, k) => {
                    filterInv.t[n][filterInv.shape[1] - 1 - i][filterInv.shape[2] - 1 - j][k] = this.weights.t[n][i][j][k];
                });
                for (let n = 0; n < N; n++) {
                    for (let f = 0; f < this.nr_filters; f++) {
                        for (let i = 0; i < h + (2 * this.padding); i++) {
                            for (let j = 0; j < w + (2 * this.padding); j++) {
                                for (let k = 0; k < f_h; k++) {
                                    for (let l = 0; l < f_w; l++) {
                                        for (let c = 0; c < ch; c++) {
                                            this.output_error.t[n][i][j][c] =
                                                this.output_error.t[n][i][j][c] + (doutp.t[n][i + k][j + l][f] * filterInv.t[f][k][l][c]);
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
    updateLayer() {
        this.optimizer.optimizeWeights();
        if (this.use_bias) {
            this.optimizer.optimizeBias();
        }
    }
    toSavedModel() {
        const data = super.toSavedModel();
        data.layer_specific = {
            nr_filters: this.nr_filters,
            filterSize: this.filterSize,
            use_bias: this.use_bias
        };
        return data;
    }
    fromSavedModel(data) {
        super.fromSavedModel(data);
        this.nr_filters = data.layer_specific.nr_filters;
        this.filterSize = data.layer_specific.filterSize;
        this.use_bias = data.layer_specific.use_bias;
    }
}

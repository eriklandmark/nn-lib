import Layer from "./layer";
import {IActivation} from "../activations/activations";
import Sigmoid from "../activations/sigmoid";
import Tensor from "../tensor";
import {SavedLayer} from "../model";

export default class DenseLayer extends Layer {

    layerSize: number

    ff_kernel: any
    act_kernel: any
    bp_error_kernel: any
    bp_error_weight_kernel: any
    weights: Tensor = new Tensor()
    errorWeights: Tensor = new Tensor()
    bias: Tensor = new Tensor()

    constructor(layerSize: number = 1, activation: IActivation = new Sigmoid()) {
        super();
        this.activationFunction = activation
        this.layerSize = layerSize;

        this.hasGPUSupport = true
        this.type = "dense"
    }

    buildLayer(prevLayerShape: number[]) {
        this.shape = [this.layerSize]
        this.prevLayerShape = prevLayerShape
        this.weights = new Tensor([prevLayerShape[0], this.layerSize], true)
        this.bias = new Tensor([1, this.layerSize], true)
        this.weights.populateRandom();
        this.bias.populateRandom();
        this.errorWeights = new Tensor()
        this.errorBias = new Tensor()
        this.output_error = new Tensor()
        this.activation = new Tensor()
    }

    /*
    buildFFKernels(batch_size: number) {
        const output_shape = [this.weights.dim().c, batch_size]
        this.ff_kernel = this.gpuInstance.createKernel(function (a, w, b) {
            let sum = 0;
            for (let i = 0; i < this.constants.arr_length; i++) {
                sum += a[this.thread.y][i] * w[i][this.thread.x];
            }
            return sum + b[this.thread.x]
        })
            .setPipeline(true)
            .setPrecision("single")
            .setConstants({arr_length: this.weights.dim().r})
            .setDynamicOutput(false)
            .setOutput(output_shape)
        this.ff_kernel.immutable = true

        this.act_kernel = this.gpuInstance.createKernel(this.activationFunction.normal_gpu())
            .setPipeline(true)
            .setConstants({softmax: this.weights.dim().c})
            .setPrecision("single")
            .setDynamicOutput(false)
            .setOutput(output_shape)
        this.act_kernel.immutable = true
    }

    buildBPKernels(length: number) {
        const output_shape = [(<Matrix>this.activation).dim().c, (<Matrix>this.activation).dim().r]
        this.bp_error_kernel = this.gpuInstance.createKernel(function (a, pW, pO) {
            let sum = 0;
            for (let i = 0; i < this.constants.mmlength; i++) {
                sum += pO[this.thread.y][i] * pW[this.thread.x][i];
            }
            // @ts-ignore
            return sum * actv_der(a[this.thread.y][this.thread.x])
        })
            .addFunction(this.activationFunction.derivative_gpu(), {output: output_shape})
            .setPipeline(true)
            .setPrecision("single")
            .setDynamicOutput(false)
            .setOutput(output_shape)
            .setConstants({mmlength: length})
        this.bp_error_kernel.immutable = true


        this.bp_error_weight_kernel = this.gpuInstance.createKernel(function (a, e) {
            let sum = 0;
            for (let i = 0; i < this.constants.arr_length; i++) {
                sum += a[i][this.thread.y] * e[i][this.thread.x];
            }
            return sum
        })
            .setPrecision("single")
            .setDynamicOutput(true)
        this.bp_error_weight_kernel.immutable = true
    }*/

    feedForward(input: Layer | Tensor, isInTraining: boolean) {
        if (this.useGpu) {
            /*const result = this.act_kernel(this.ff_kernel(input, this.weights.toNumberArray(), this.bias.toNumberArray()))
            this.activation = new Matrix(result.toArray())
            return result*/
        } else {
            let act: Tensor = input instanceof Tensor? input: (<Layer>input).activation
            const z = act.dot(this.weights)
            z.iterate((pos) => {z.set(pos, z.get(pos) + this.bias.t[0][pos[1]])}, true)
            this.activation = <Tensor>this.activationFunction.normal(z)
        }
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor) {
        if (this.useGpu) {
            /*let input: Matrix
            if (next_layer instanceof Layer) {
                input = <Matrix>next_layer.activation
            } else {
                input = next_layer
            }
            const error = this.bp_error_kernel(
                (<Matrix> this.activation).toNumberArray(),
                (<Matrix>prev_layer.weights).toNumberArray(),
                prev_layer.output_error)
            this.output_error = error
            this.bp_error_weight_kernel.setOutput([(<Matrix>this.activation).dim().c, input.dim().c])
                .setConstants({arr_length: input.dim().r})
            const error_weights = this.bp_error_weight_kernel(input.toNumberArray(), error)
            this.errorWeights = new Matrix(error_weights)
            const errorMatrix = new Matrix(error.toArray())
            this.errorBias = <Matrix>errorMatrix.sum(0)*/
        } else {
            let dzh_dwh: Tensor = next_layer instanceof Layer ? next_layer.activation: next_layer
            const error = prev_layer.output_error.dot(prev_layer.weights.transpose())
                .mul(this.activationFunction.derivative(this.activation))
            this.errorWeights = dzh_dwh.transpose().dot(error);
            this.errorBias = <Tensor> error.sum(0)
            this.output_error = error;
        }
    }

    toSavedModel(): SavedLayer {
        const data = super.toSavedModel();
        data.layer_specific.layerSize = this.layerSize
        return data
    }

    fromSavedModel(data: SavedLayer) {
        this.layerSize = data.layer_specific.layerSize
        super.fromSavedModel(data);
    }
}
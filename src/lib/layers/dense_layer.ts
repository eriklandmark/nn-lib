import Layer from "./layer";
import Matrix from "../../matrix";
import Activation, {IActivation} from "../activations/activations";
import Vector from "../../vector";
import {SavedLayer} from "../../model";
import Sigmoid from "../activations/sigmoid";

export default class DenseLayer extends Layer {

    layerSize: number

    ff_kernel: any
    act_kernel: any
    bp_error_kernel: any
    bp_error_weight_kernel: any

    constructor(layerSize: number = 1, activation: IActivation = new Sigmoid()) {
        super();
        this.activationFunction = activation
        this.layerSize = layerSize;

        this.type = "dense"
    }

    buildLayer(prevLayerShape: number[]) {
        this.shape = [this.layerSize]
        this.prevLayerShape = prevLayerShape
        this.weights = new Matrix()
        this.weights.createEmptyArray(prevLayerShape[0], this.layerSize)
        this.bias = new Vector(this.layerSize)
        this.weights.populateRandom();
        this.bias.populateRandom();
        this.errorWeights = new Matrix()
        this.errorBias = new Matrix()
        this.output_error = new Matrix()
        this.activation = new Matrix()
    }

    buildFFKernels(batch_size) {
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
    }

    feedForward(input: Layer | Matrix, isInTraining: boolean, gpu: boolean = false) {
        if (gpu) {
            const result = this.act_kernel(this.ff_kernel(input, this.weights.toNumberArray(), this.bias.toNumberArray()))
            this.activation = new Matrix(result.toArray())
            return result
        } else {
            let act: Matrix
            if (input instanceof Matrix) {
                act = input
            } else {
                act = <Matrix>(<Layer>input).activation
            }
            const z = <Matrix>act.mm(this.weights)
            z.iterate((i: number, j: number) => {
                z.set(i, j, z.get(i, j) + this.bias.get(j))
            })
            this.activation = <Matrix>this.activationFunction.normal(z)
        }
    }

    calculate_errors(error: any, input: Matrix) {

    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix, gpu: boolean) {
        if (gpu) {
            let input: Matrix
            if (next_layer instanceof Layer) {
                input = <Matrix>next_layer.activation
            } else {
                input = next_layer
            }
            const error = this.bp_error_kernel(
                (<Matrix> this.activation).toNumberArray(),
                prev_layer.weights.toNumberArray(),
                prev_layer.output_error)
            this.output_error = error
            this.bp_error_weight_kernel.setOutput([(<Matrix>this.activation).dim().c, input.dim().c])
                .setConstants({arr_length: input.dim().r})
            const error_weights = this.bp_error_weight_kernel(input.toNumberArray(), error)
            this.errorWeights = new Matrix(error_weights)
            const errorMatrix = new Matrix(error.toArray())
            this.errorBias = <Matrix>errorMatrix.sum(0)
        } else {
            let dzh_dwh: Matrix
            if (next_layer instanceof Layer) {
                dzh_dwh = <Matrix>next_layer.activation
            } else {
                dzh_dwh = next_layer
            }
            const deltaActv = <Matrix>this.activationFunction.derivative(<Matrix>this.activation)
            // @ts-ignore
            const error = ((<Matrix>prev_layer.output_error).mm(prev_layer.weights.transpose())).mul(deltaActv)
            this.errorWeights = <Matrix>dzh_dwh.transpose().mm(error);
            this.errorBias = <Matrix>error.sum(0)
            this.output_error = error;
        }
    }

    updateWeights(l_rate: number) {
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate))
        this.bias.iterate((val: number, i: number) => {
            this.bias.set(i, val - (this.errorBias.get(0, i) * l_rate))
        })
    }

    toSavedModel(): SavedLayer {
        return {
            weights: this.weights.matrix,
            bias: this.bias.vector,
            shape: this.shape,
            activation: this.activationFunction.name
        }
    }

    fromSavedModel(data: SavedLayer) {
        this.weights = Matrix.fromJsonObject(data.weights)
        this.bias = Vector.fromJsonObj(data.bias)
        this.shape = data.shape
        this.activationFunction = Activation.fromName(data.activation)
    }
}
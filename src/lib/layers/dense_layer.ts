import Layer from "./layer";
import Matrix from "../../matrix";
import IActivation from "../activations/activations";
import Vector from "../../vector";

export default class DenseLayer extends Layer {

    layerSize: number

    constructor(layerSize: number, activation: IActivation) {
        super(activation);
        this.layerSize = layerSize;
    }

    buildLayer(prevLayerShape: number[]) {
        this.shape = [this.layerSize]

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

    feedForward(input: Layer | Matrix, isInTraining: boolean) {
        let act: Matrix
        if (input instanceof Matrix) {
            act = input
        } else {
            act = <Matrix>(<Layer>input).activation
        }

        if (this.useGpu) {
            /*
            const ffKernel = this.gpuInstance.createKernelMap({
                addResult: Matrix.addGpu(),
                multiplyResult: Matrix.mmGpu(),
                actvResult: this.activationFunction.normal_gpu()
            }, function (a: ThreadKernelVariable, b: ThreadKernelVariable, c: ThreadKernelVariable) {
                //@ts-ignore
                return actv(add(mm(a, b), c[this.thread.x]));
            }, {output: [this.weights.dim().c, act.dim().r], constants: {mmLength: act.dim().c}})
            ffKernel.setLoopMaxIterations(Math.max(act.dim().c, this.weights.dim().r))
            this.activation = new Matrix(<Float32Array[]>ffKernel(act.toNumberArray(), this.weights.toNumberArray(), this.bias.toNumberArray())["result"]);
            ffKernel.destroy()*/
        } else {
            const z = <Matrix>act.mm(this.weights)
            z.iterate((i: number, j: number) => {
                z.set(i, j, z.get(i, j) + this.bias.get(j))
            })
            this.activation = <Matrix>this.activationFunction.normal(z)
        }
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        let dzh_dwh: Matrix
        if (next_layer instanceof Layer) {
            dzh_dwh = <Matrix>next_layer.activation
        } else {
            dzh_dwh = next_layer
        }
        /*
        const feedForwardKernel = gpu.createKernelMap({
            addResult: Matrix.addGpu(),
            multiplyResult: Matrix.mmGpu(),
            actvResult: Activations.sigmoid_gpu()
        }, function(a, b, c) {
            //@ts-ignore
            return actv(add(mm(a, b), c[this.thread.y][this.thread.x]));
        }, { output: [b.dim().c, a.dim().r], constants: {mmLength: a.dim().c}})
        feedForwardKernel.setLoopMaxIterations(Math.max(a.dim().c, b.dim().r))


        new Matrix(<Float32Array[]>feedForwardKernel(a.toNumberArray(), b.toNumberArray(), c.toNumberArray()).result)
        */

        const deltaActv = <Matrix> this.activationFunction.derivative(<Matrix>this.activation)
        // @ts-ignore
        const error = ((<Matrix>prev_layer.output_error).mm(prev_layer.weights.transpose())).mul(deltaActv)
        this.errorWeights = <Matrix>dzh_dwh.transpose().mm(error);
        this.errorBias = <Matrix>error.sum(0)
        this.output_error = error;
    }

    updateWeights(l_rate: number) {
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate))
        this.bias.iterate((val: number, i: number) => {
            this.bias.set(i, val - (this.errorBias.get(0, i) * l_rate))
        })
    }
}
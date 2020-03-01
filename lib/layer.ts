import Matrix from "./matrix";
import Vector from "./vector";
import {GPU} from "gpu.js";
import Activations from "./activations";

export default class Layer {
    weights: Matrix
    bias: Vector
    errorWeights: Matrix
    errorBias: Matrix
    output_error: Matrix
    activation: Matrix
    actFunc: Function
    actFuncDer: Function
    layerSize: number
    useGpu: boolean = false;
    gpuInstance: GPU

    constructor(layerSize: number, prevLayerSize: number, actFunc: Function, actFuncDer: Function) {
        this.layerSize = layerSize
        this.weights = new Matrix()
        this.weights.createEmptyArray(prevLayerSize, layerSize)
        this.bias = new Vector(layerSize)
        this.errorWeights = new Matrix()
        this.errorBias = new Matrix()
        this.output_error = new Matrix()
        this.activation = new Matrix()
        this.actFunc = actFunc
        this.actFuncDer = actFuncDer
    }

    setGpuInstance(gpuIns: GPU) {
        this.gpuInstance = gpuIns;
    }

    public populate() {
        this.weights.populateRandom();
        this.bias.populateRandom();
    }

    public feedForward(input: Layer | Matrix) {
        let act
        if (input instanceof Matrix) {
            if (this.activation.empty()) {
                this.activation.createEmptyArray(input.dim().r, this.layerSize)
            }
            act = input
        } else {
            if (this.activation.empty()) {
                this.activation.createEmptyArray(input.activation.dim().r, this.layerSize)
            }

            act = (<Layer>input).activation
        }
        if (this.useGpu) {
            const ffKernel = this.gpuInstance.createKernelMap({
                addResult: Matrix.addGpu(),
                multiplyResult: Matrix.mmGpu(),
                actvResult: Activations.sigmoid_gpu()
            }, function (a, b, c) {
                //@ts-ignore
                return actv(add(mm(a, b), c[this.thread.x]));
            }, {output: [this.weights.dim().c, act.dim().r], constants: {mmLength: act.dim().c}})
            ffKernel.setLoopMaxIterations(Math.max(act.dim().c, this.weights.dim().r))
            this.activation = new Matrix(<Float32Array[]>ffKernel(act.toNumberArray(), this.weights.toNumberArray(), this.bias.toNumberArray())["result"]);
            ffKernel.destroy()
        } else {
            const z = <Matrix>act.mm(this.weights)
            z.iterate((i, j) => {
                z.set(i, j, z.get(i, j) + this.bias.get(j))
            })
            this.activation = this.actFunc(z)
        }
    }

    updateWeights(l_rate: number) {
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate))
        this.bias.iterate((val: number, i: number) => {
            this.bias.set(i, val - (this.errorBias.get(0, i) * l_rate))
        })
    }
}
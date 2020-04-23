import Matrix from "../../matrix";
import Vector from "../../vector";
import {GPU} from "gpu.js";
import {IActivation} from "../activations/activations";
import Tensor from "../../tensor";
import {SavedLayer} from "../../model";

export default class Layer {
    weights: Matrix = new Matrix()
    bias: Vector = new Vector()
    errorWeights: Matrix = new Matrix()
    errorBias: Matrix | Vector = new Matrix()
    output_error: any
    activation: Matrix | Tensor[] = new Matrix()
    activationFunction: IActivation
    useGpu: boolean = false;
    gpuInstance: GPU = new GPU()
    shape: number[] = []
    prevLayerShape: number[] = []
    type: string = ""
    hasGPUSupport = false;
    isFirstLayer = false;

    ff_kernel: any
    act_kernel: any
    bp_error_kernel: any
    bp_error_weight_kernel: any

    setGpuInstance(gpuIns: GPU) {
        this.gpuInstance = gpuIns;
    }

    getLayerInfo() {
        return {
            type: this.type,
            shape: this.shape,
            activation: this.activationFunction ? this.activationFunction.name : "NO ACTIVATION"
        }
    }

    buildLayer(prevLayerShape: number[]) {}
    feedForward(input: Layer | Matrix | Tensor[], isInTraining: boolean) {}
    buildFFKernels(batch_size: number) {}
    buildBPKernels(size: number) {}
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix | Tensor[]) {}
    calculate_errors(error: any, next_layer: Layer | Matrix) {}
    updateWeights(l_rate: number) {}
    toSavedModel(): SavedLayer {return}
    fromSavedModel(data: SavedLayer) {}
}
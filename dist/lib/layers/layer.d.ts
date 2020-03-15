import Matrix from "../../matrix";
import Vector from "../../vector";
import {GPU} from "gpu.js";
import {IActivation} from "../activations/activations";
import Tensor from "../../tensor";
import {SavedLayer} from "../../model";

export default class Layer {
    weights: Matrix;
    bias: Vector;
    errorWeights: Matrix;
    errorBias: Matrix | Vector;
    output_error: Matrix | Tensor[];
    activation: Matrix | Tensor[];
    activationFunction: IActivation;
    useGpu: boolean;
    gpuInstance: GPU;
    shape: number[];
    type: string;
    setGpuInstance(gpuIns: GPU): void;
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Matrix | Tensor[], isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix | Tensor[]): void;
    updateWeights(l_rate: number): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}

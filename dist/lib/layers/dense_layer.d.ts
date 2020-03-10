import Layer from "./layer";
import Matrix from "../../matrix";
import IActivation from "../activations/activations";

export default class DenseLayer extends Layer {
    layerSize: number;
    constructor(layerSize: number, activation: IActivation);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Matrix, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
    updateWeights(l_rate: number): void;
}

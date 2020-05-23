import Layer from "./layer";
import DenseLayer from "./dense_layer";
import { IActivation } from "../activations/activations";
import { ILoss } from "../losses/losses";
import { SavedLayer } from "../model";
import { IGradient } from "../losses/gradients";
import Tensor from "../tensor";
export default class OutputLayer extends DenseLayer {
    loss: number;
    accuracy: number;
    layerSize: number;
    lossFunction: ILoss;
    gradientFunction: IGradient;
    constructor(layerSize?: number, activation?: IActivation);
    buildLayer(prevLayerShape: number[]): void;
    backPropagationOutputLayer(labels: Tensor, next_layer: Layer): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}

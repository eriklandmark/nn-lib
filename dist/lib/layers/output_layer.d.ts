import Layer from "./layer";
import Matrix from "../../matrix";
import DenseLayer from "./dense_layer";
import { IActivation } from "../activations/activations";
import { ILoss } from "../losses/losses";
import { SavedLayer } from "../../model";
import { IGradient } from "../losses/gradients";
export default class OutputLayer extends DenseLayer {
    loss: number;
    layerSize: number;
    lossFunction: ILoss;
    gradientFunction: IGradient;
    constructor(layerSize?: number, activation?: IActivation);
    buildLayer(prevLayerShape: number[]): void;
    backPropagationOutputLayer(labels: Matrix, next_layer: Layer): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}

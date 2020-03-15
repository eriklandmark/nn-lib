import Layer from "./layer";
import Matrix from "../../matrix";
import DenseLayer from "./dense_layer";
import {IActivation} from "../activations/activations";
import {ILoss} from "../losses/losses";
import {SavedLayer} from "../../model";

export default class OutputLayer extends DenseLayer {
    loss: number;
    layerSize: number;
    lossFunction: ILoss;
    constructor(layerSize?: number, activation?: IActivation);
    backPropagationOutputLayer(labels: Matrix, next_layer: Layer): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}

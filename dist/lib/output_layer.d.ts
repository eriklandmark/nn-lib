import Layer from "./layer";
import Matrix from "./matrix";

export default class DenseLayer extends Layer {
    loss: number;
    lossFunction: Function;
    backPropagation(labels: Matrix, next_layer: Layer): void;
}

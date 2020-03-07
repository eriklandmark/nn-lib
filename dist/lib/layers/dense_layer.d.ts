import Layer from "./layer";
import Matrix from "../../matrix";

export default class DenseLayer extends Layer {
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
}

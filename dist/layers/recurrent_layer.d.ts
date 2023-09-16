import Layer from "./layer";
import Tensor from "../tensor";
export default class RecurrentLayer extends Layer {
    hidden_size: number;
    vocab_size: number;
    U: Tensor;
    V: Tensor;
    weights: Tensor;
    bias: Tensor;
    bias_hidden: Tensor;
    constructor(hidden_size: number, vocab_size: number);
}

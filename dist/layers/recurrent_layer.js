import Layer from "./layer";
import Tensor from "../tensor";
export default class RecurrentLayer extends Layer {
    constructor(hidden_size, vocab_size) {
        super();
        this.U = new Tensor();
        this.V = new Tensor();
        this.weights = new Tensor();
        this.bias = new Tensor();
        this.bias_hidden = new Tensor();
        this.hidden_size = hidden_size;
        this.vocab_size = vocab_size;
    }
}

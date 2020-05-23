import Layer from "./layer";
import Tensor from "../tensor";

export default class RecurrentLayer extends Layer {

    hidden_size: number;
    vocab_size: number
    U: Tensor = new Tensor()
    V: Tensor = new Tensor()
    weights = new Tensor()
    bias = new Tensor()
    bias_hidden = new Tensor()

    constructor(hidden_size: number, vocab_size: number) {
        super();
        this.hidden_size = hidden_size
        this.vocab_size = vocab_size
    }

    /*buildLayer(prevLayerShape: number[]) {
        this.shape = [this.hidden_size, this.vocab_size]
        this.prevLayerShape = prevLayerShape
        this.U = new Tensor(this.hidden_size, this.vocab_size)
        this.V.createEmptyArray(this.hidden_size, this.hidden_size)
        this.weights.createEmptyArray(this.vocab_size, this.hidden_size)
        this.bias.createEmptyArray(this.vocab_size, 1)
        this.bias_hidden.createEmptyArray(this.hidden_size, 1)
        this.U.populateRandom()
        this.V.populateRandom()
        this.weights.populateRandom();
        this.bias.populateRandom();
        this.bias_hidden.populateRandom()
        this.errorWeights = new Matrix()
        this.errorBias = new Matrix()
        this.output_error = new Matrix()
        this.activation = new Matrix()
    }*/
}
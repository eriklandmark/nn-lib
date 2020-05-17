import Layer from "./layer";
import Matrix from "../matrix";

export default class RecurrentLayer extends Layer {

    hidden_size: number;
    vocab_size: number
    U: Matrix = new Matrix()
    V: Matrix = new Matrix()
    weights = new Matrix()
    bias = new Matrix()
    bias_hidden = new Matrix()

    constructor(hidden_size: number, vocab_size: number) {
        super();
        this.hidden_size = hidden_size
        this.vocab_size = vocab_size
    }

    buildLayer(prevLayerShape: number[]) {
        this.shape = [this.hidden_size, this.vocab_size]
        this.prevLayerShape = prevLayerShape
        this.U.createEmptyArray(this.hidden_size, this.vocab_size)
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
    }
}
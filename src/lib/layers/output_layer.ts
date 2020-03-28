import Layer from "./layer";
import Matrix from "../../matrix";
import DenseLayer from "./dense_layer";
import Activation, {IActivation} from "../activations/activations";
import Losses, {ILoss} from "../losses/losses";
import MeanSquaredError from "../losses/mean_squared_error";
import {SavedLayer} from "../../model";
import Vector from "../../vector";
import Sigmoid from "../activations/sigmoid";

export default class OutputLayer extends DenseLayer {

    loss: number = 0;
    layerSize: number = 0;
    lossFunction: ILoss = new MeanSquaredError()

    constructor(layerSize: number = 1, activation: IActivation = new Sigmoid()) {
        super(layerSize, activation)
        this.layerSize = layerSize
        this.type = "output"
    }

    public backPropagationOutputLayer(labels: Matrix, next_layer: Layer) {
        this.loss = <number> labels.mul(-1).mul((<Matrix> this.activation).log()).sum()
        const gradient = <Matrix> this.lossFunction.derivative((<Matrix> this.activation), labels)
        this.errorBias = gradient
        this.output_error = gradient

        this.errorWeights = <Matrix> (<Matrix> next_layer.activation).transpose().mm(gradient)
    }

    toSavedModel(): SavedLayer {
        return {
            weights: this.weights.matrix,
            bias: this.bias.vector,
            loss: this.lossFunction.name,
            shape: this.shape,
            activation: this.activationFunction.name
        }
    }

    fromSavedModel(data: SavedLayer) {
        this.weights = Matrix.fromJsonObject(data.weights)
        this.bias = Vector.fromJsonObj(data.bias)
        this.shape = data.shape
        this.activationFunction = Activation.fromName(data.activation)
        this.lossFunction = Losses.fromName(data.loss)
    }
}
import Layer from "./layer";
import Matrix from "../matrix";
import DenseLayer from "./dense_layer";
import Activation, {IActivation} from "../activations/activations";
import Losses, {ILoss} from "../losses/losses";
import {SavedLayer} from "../model";
import Vector from "../vector";
import Sigmoid from "../activations/sigmoid";
import Gradients, {IGradient} from "../losses/gradients";

export default class OutputLayer extends DenseLayer {

    loss: number = 0;
    accuracy: number = 0;
    layerSize: number = 0;
    lossFunction: ILoss
    gradientFunction: IGradient

    constructor(layerSize: number = 1, activation: IActivation = new Sigmoid()) {
        super(layerSize, activation)
        this.layerSize = layerSize
        this.type = "output"
    }

    buildLayer(prevLayerShape: number[]) {
        super.buildLayer(prevLayerShape);
        this.gradientFunction = Gradients.get_gradient(this.activationFunction, this.lossFunction)
    }

    public backPropagationOutputLayer(labels: Matrix, next_layer: Layer) {
        this.loss = <number> labels.mul(-1).mul((<Matrix> this.activation).log()).sum()
        const gradient = this.gradientFunction((<Matrix> this.activation), labels)
        let total_acc = 0
        let total_loss = 0
        for (let i = 0; i < labels.dim().r; i++) {
            total_acc += (<Matrix> this.activation).argmax(i) == labels.argmax(i)? 1:0
            total_loss += Math.abs(gradient.get(i,0))
        }
        this.accuracy = total_acc / labels.dim().r
        //this.loss = total_loss

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
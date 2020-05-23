import Layer from "./layer";
import DenseLayer from "./dense_layer";
import Activation, {IActivation} from "../activations/activations";
import Losses, {ILoss} from "../losses/losses";
import {SavedLayer} from "../model";
import Sigmoid from "../activations/sigmoid";
import Gradients, {IGradient} from "../losses/gradients";
import Tensor from "../tensor";

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

    public backPropagationOutputLayer(labels: Tensor, next_layer: Layer) {
        this.loss = <number> labels.mul(-1).mul((<Tensor> this.activation).add(10**-8).log()).sum() / labels.shape[0]
        const gradient = this.gradientFunction((<Tensor> this.activation), labels)
        let total_acc = 0
        for (let i = 0; i < labels.shape[0]; i++) {
            total_acc += (<Tensor> this.activation).argmax(i) == labels.argmax(i)? 1:0
        }
        this.accuracy = total_acc / labels.shape[0]

        this.errorBias = <Tensor> gradient.sum(0)
        this.output_error = gradient
        this.errorWeights = <Tensor> next_layer.activation.transpose().dot(gradient)
    }

    toSavedModel(): SavedLayer {
        const data = super.toSavedModel()
        data.layer_specific = {
            layerSize: this.layerSize,
            loss: this.lossFunction? this.lossFunction.name: "cross_entropy"
        }

        return data
    }

    fromSavedModel(data: SavedLayer) {
        super.fromSavedModel(data)
        this.layerSize = data.layer_specific.layerSize
        this.lossFunction = new (Losses.fromName(data.layer_specific.loss))()
    }
}
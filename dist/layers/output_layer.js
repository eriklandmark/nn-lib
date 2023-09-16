import DenseLayer from "./dense_layer";
import Losses from "../losses/losses";
import Sigmoid from "../activations/sigmoid";
import Gradients from "../losses/gradients";
export default class OutputLayer extends DenseLayer {
    constructor(layerSize = 1, activation = new Sigmoid()) {
        super(layerSize, activation);
        this.loss = 0;
        this.accuracy = 0;
        this.layerSize = 0;
        this.layerSize = layerSize;
        this.type = "output";
    }
    buildLayer(prevLayerShape) {
        super.buildLayer(prevLayerShape);
        this.gradientFunction = Gradients.get_gradient(this.activationFunction, this.lossFunction);
    }
    backPropagationOutputLayer(labels, next_layer) {
        this.loss = labels.mul(-1).mul(this.activation.add(Math.pow(10, -8)).log()).sum() / labels.shape[0];
        const gradient = this.gradientFunction(this.activation, labels);
        let total_acc = 0;
        for (let i = 0; i < labels.shape[0]; i++) {
            total_acc += this.activation.argmax(i) == labels.argmax(i) ? 1 : 0;
        }
        this.accuracy = total_acc / labels.shape[0];
        this.errorBias = gradient.sum(0);
        this.output_error = gradient;
        this.errorWeights = next_layer.activation.transpose().dot(gradient);
    }
    toSavedModel() {
        const data = super.toSavedModel();
        data.layer_specific = {
            layerSize: this.layerSize,
            loss: this.lossFunction ? this.lossFunction.name : "cross_entropy"
        };
        return data;
    }
    fromSavedModel(data) {
        super.fromSavedModel(data);
        this.layerSize = data.layer_specific.layerSize;
        this.lossFunction = new (Losses.fromName(data.layer_specific.loss))();
    }
}

import Layer from "./layer";
import Vector from "./vector";
import Activations from "./activations";

export default class DenseLayer extends Layer{

    public backPropagation(prev_layer: Layer) {
        const dEo1dOuth2 = prev_layer.output_error.mul(prev_layer.dActivations)
        const dEtotdOuth2 = <Vector> prev_layer.weights.transpose().mm(dEo1dOuth2)
        this.dActivations = <Vector> Activations.sigmoid_derivative(this.activation)
        const deltaError = dEtotdOuth2.mul(this.dActivations).mul(this.activation)
        this.error = deltaError;
        this.output_error = deltaError;
    }
}
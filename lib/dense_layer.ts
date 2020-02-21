import Layer from "./layer";
import Vector from "./vector";
import Activations from "./activations";
import Matrix from "./matrix";

export default class DenseLayer extends Layer{

    public backPropagationOld(prev_layer: Layer) {
        this.dActivations = <Vector> Activations.sigmoid_derivative(this.activation)
        console.log(this.dActivations.toString())
        const mat = new Matrix([prev_layer.output_error]).transpose()
        console.log(prev_layer.weights.toString())
        const deltaError = prev_layer.weights.transpose().mm(mat)
        console.log(deltaError.toString())
        //this.error = deltaError;
        //this.output_error = deltaError;
    }

    public backPropagation(prev_layer: Layer) {
        const dEo1dOuth2 = prev_layer.output_error.mul(prev_layer.dActivations)
        const dEtotdOuth2 = <Vector> prev_layer.weights.transpose().mm(dEo1dOuth2)
        this.dActivations = <Vector> Activations.ReLu_derivative(this.activation)
        const deltaError = dEtotdOuth2.mul(this.dActivations).mul(this.activation)
        this.error = deltaError;
        this.output_error = deltaError;
    }
}
import Layer from "./layer";
import Vector from "./vector";
import Losses from "./losses";
import Activations from "./activations";

export default class DenseLayer extends Layer{

    totalError: number = 0;

    public backPropagation(labels: Vector) {
        const errorVector = <Vector> Losses.squared_error(this.activation, labels)
        this.totalError = errorVector.sum()

        const dError = <Vector> Losses.squared_error_derivative(this.activation, labels)
        this.dActivations = <Vector> Activations.sigmoid_derivative(this.activation)
        this.error = dError.mul(this.dActivations).mul(this.activation)
        this.output_error = this.error.mul(this.activation)
    }
}
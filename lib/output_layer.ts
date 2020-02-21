import Layer from "./layer";
import Vector from "./vector";
import Losses from "./losses";
import Matrix from "./matrix";
import Activations from "./activations";

export default class DenseLayer extends Layer{

    totalError: number = 0;

    /*public backPropagation(labels: Vector, next_layer: Layer) {
        const errorVector = <Vector> Losses.CrossEntropy(this.activation, labels)
        this.totalError = errorVector.sum()

        const dA = new Matrix([Losses.CrossEntropy_derivative(this.activation, labels)]).transpose()
        this.dActivations = <Vector> Activations.sigmoid_derivative(this.activation)
        const dZ = dJ.mul(this.dActivations)
        this.error = dJ.mul(this.dActivations).mul(this.activation)
        this.output_error = this.error
    }*/

    public backPropagation(labels: Vector) {
        const errorVector = <Vector> Losses.CrossEntropy(this.activation, labels)
        this.totalError = errorVector.sum()

        const dError = <Vector> Losses.CrossEntropy_derivative(this.activation, labels)
        this.dActivations = <Vector> Activations.ReLu_derivative(this.activation)
        this.error = dError.mul(this.dActivations).mul(this.activation)
        this.output_error = this.error.mul(this.activation)
    }
}
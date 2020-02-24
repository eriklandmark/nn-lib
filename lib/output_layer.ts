import Layer from "./layer";
import Losses from "./losses";
import Matrix from "./matrix";

export default class DenseLayer extends Layer{

    loss: number = 0;

    public backPropagation(labels: Matrix) {

        this.errorBias = <Matrix> Losses.squared_error_derivative(this.activation, labels)
        //this.loss = <number> labels.mul(-1).mul(this.activation.log()).sum()
        this.errorWeights = <Matrix> this.activation.transpose().mm(this.errorBias)
        this.output_error = this.errorBias
    }
    /*
    public backPropagationOld(labels: Vector) {
        const errorVector = <Vector> Losses.squared_error(this.activation, labels)
        this.totalError = errorVector.sum()

        const dError = <Vector> Losses.squared_error_derivative(this.activation, labels)
        this.dActivations = <Vector> Activations.Softmax(this.activation)
        this.error = dError.mul(this.dActivations).mul(this.activation)
        this.output_error = this.error.mul(this.activation)
    }*/
}
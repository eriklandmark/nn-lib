import Layer from "./layer";
import Losses from "./losses";
import Matrix from "./matrix";

export default class DenseLayer extends Layer{

    loss: number = 0;

    public backPropagation(labels: Matrix, next_layer: Layer) {
        const gradient = <Matrix> Losses.squared_error_derivative(this.activation, labels).mul(2)
        //console.log(gradient)
        this.loss = <number> labels.mul(this.activation.log()).mul(-1).sum()
        const error = <Matrix> gradient//.mul(this.actFuncDer(this.activation))
        this.errorWeights = <Matrix> next_layer.activation.transpose().mm(error)
        this.output_error = error
    }

    public backPropagationOld(labels: Matrix, next_layer: Layer) {

        this.errorBias = <Matrix> Losses.squared_error_derivative(this.activation, labels)
        //console.log(this.errorBias.toString())
        this.loss = <number> labels.mul(-1).mul(this.activation.log()).sum()
        this.errorWeights = <Matrix> next_layer.activation.transpose().mm(this.errorBias)
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
import Layer from "./layer";
import Losses from "./losses";
import Matrix from "./matrix";

export default class DenseLayer extends Layer{

    loss: number = 0;

    public backPropagation(labels: Matrix, next_layer: Layer) {
        const gradient = <Matrix> Losses.squared_error_derivative(this.activation, labels)
        this.loss = <number> labels.mul(this.activation.log()).mul(-1).sum()
        const error = <Matrix> gradient//.mul(this.actFuncDer(this.activation))
        this.errorWeights = <Matrix> next_layer.activation.transpose().mm(error)
        this.output_error = error
    }
}
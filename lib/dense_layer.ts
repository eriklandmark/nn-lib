import Layer from "./layer";
import Matrix from "./matrix";

export default class DenseLayer extends Layer {

    public backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        let dzh_dwh: Matrix
        if (next_layer instanceof Layer) {
            dzh_dwh = next_layer.activation
        } else {
            dzh_dwh = next_layer
        }

        const error = (<Matrix>prev_layer.output_error.mm(prev_layer.weights.transpose())).mul(this.actFuncDer(this.activation))
        this.errorWeights = <Matrix>dzh_dwh.transpose().mm(error);
        this.errorBias = <Matrix> error.sum(0)
        this.output_error = error;
    }
}
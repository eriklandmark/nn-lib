import Layer from "./layer";
import Vector from "./vector";
import Activations from "./activations";
import Matrix from "./matrix";

export default class DenseLayer extends Layer{

    /*public backPropagationOld(prev_layer: Layer) {
        this.dActivations = <Vector> Activations.sigmoid_derivative(this.activation)
        console.log(this.dActivations.toString())
        const mat = new Matrix([prev_layer.output_error]).transpose()
        console.log(prev_layer.weights.toString())
        const deltaError = prev_layer.weights.transpose().mm(mat)
        console.log(deltaError.toString())
        //this.error = deltaError;
        //this.output_error = deltaError;
    }*/

    public backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        const dcost_dah = <Matrix> prev_layer.output_error.mm(prev_layer.weights.transpose())
        let dzh_dwh: Matrix

        if (next_layer instanceof Layer) {
            dzh_dwh = next_layer.z
        } else {
            dzh_dwh = next_layer
        }

        const dah_dzh = <Matrix> this.actFuncDer(prev_layer.activation)
        //console.log(dah_dzh.toString())
        this.errorWeights = dzh_dwh.mul(dah_dzh.mul(dcost_dah));
        this.errorBias = dcost_dah.mul(dah_dzh)
        this.output_error = this.errorWeights;
    }
}
import Layer from "./layer";
import Vector from "./vector";
import Activations from "./activations";
import Matrix from "./matrix";

export default class DenseLayer extends Layer{

    public backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        const dcost_dah = <Matrix> prev_layer.output_error.mm(prev_layer.weights.transpose())
        let dzh_dwh: Matrix

        if (next_layer instanceof Layer) {
            dzh_dwh = next_layer.activation
        } else {
            dzh_dwh = next_layer
        }

        const dah_dzh = <Matrix> this.actFuncDer(this.activation)
        const error = dcost_dah.mul(dah_dzh)
        this.errorWeights = <Matrix> dzh_dwh.transpose().mm(error);
        this.output_error = error;
    }

    public backPropagationOld(prev_layer: Layer, next_layer: Layer | Matrix) {
        console.log("Prev Layer Out: ", prev_layer.output_error.dim())
        console.log("Prev Layer we:", prev_layer.weights.dim())
        const dcost_dah = <Matrix> prev_layer.output_error.mm(prev_layer.weights.transpose())
        let dzh_dwh: Matrix

        if (next_layer instanceof Layer) {
            dzh_dwh = next_layer.activation
        } else {
            console.log("hej")
            dzh_dwh = next_layer
        }

        console.log("dzh_dwh",dzh_dwh.dim())

        const dah_dzh = <Matrix> this.actFuncDer(this.activation)
        console.log(dah_dzh.dim(), dcost_dah.dim())
        const ss = dah_dzh.mul(dcost_dah)
        this.errorWeights = <Matrix> dzh_dwh.transpose().mm(ss);
        console.log("error w",this.errorWeights.dim())
        //this.errorBias = dcost_dah.mul(dah_dzh)
        this.output_error = this.errorWeights;
        console.log("-----")
    }
}
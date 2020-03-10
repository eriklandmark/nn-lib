import Layer from "./layer";
import Matrix from "../../matrix";

export default class DenseLayer extends Layer {

    public backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        let dzh_dwh: Matrix
        if (next_layer instanceof Layer) {
            dzh_dwh = next_layer.activation
        } else {
            dzh_dwh = next_layer
        }
        /*
        const feedForwardKernel = gpu.createKernelMap({
            addResult: Matrix.addGpu(),
            multiplyResult: Matrix.mmGpu(),
            actvResult: Activations.sigmoid_gpu()
        }, function(a, b, c) {
            //@ts-ignore
            return actv(add(mm(a, b), c[this.thread.y][this.thread.x]));
        }, { output: [b.dim().c, a.dim().r], constants: {mmLength: a.dim().c}})
        feedForwardKernel.setLoopMaxIterations(Math.max(a.dim().c, b.dim().r))


        new Matrix(<Float32Array[]>feedForwardKernel(a.toNumberArray(), b.toNumberArray(), c.toNumberArray()).result)
        */

        const error = (<Matrix>prev_layer.output_error.mm(prev_layer.weights.transpose())).mul(this.actFuncDer!(this.activation))
        this.errorWeights = <Matrix>dzh_dwh.transpose().mm(error);
        this.errorBias = <Matrix>error.sum(0)
        this.output_error = error;
    }
}
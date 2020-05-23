import Layer from "./layer";

export default class BatchNormLayer extends Layer {

    /*
    momentum: number
    running_mean: Matrix
    running_var: Matrix
    cache: any = {}

    weights: Matrix
    errorWeights: Matrix

    constructor(momentum: number = 0.9) {
        super();
        this.momentum = momentum
        this.type = "batch_norm"
    }

    buildLayer(prevLayerShape: number[]) {
        const [D] = prevLayerShape
        console.log(prevLayerShape)
        this.shape = prevLayerShape
        this.running_mean = new Matrix()
        this.running_mean.createEmptyArray(1,D)
        this.running_var = new Matrix()
        this.running_var.createEmptyArray(1,D)
        this.weights = new Matrix()
        this.weights.createEmptyArray(1,D)
        this.weights.populateRandom()
        this.bias = new Matrix()
        this.bias.createEmptyArray(1,D)
        this.bias.populateRandom()
    }

    feedForward(input: Layer | Matrix, isInTraining: boolean) {
        let act: Matrix
        if (input instanceof Matrix) {
            act = input
        } else {
            act = <Matrix>(<Layer>input).activation
        }
        const N = act.dim().r
        const mean = <Matrix> act.mean(0)
        const diff = act.sub(mean.repeat(0, N))
        const variance = <Matrix> diff.pow(2).mean(0)
        if(isInTraining) {
            console.log(act.toString())
            const xhat = diff.div(variance.sqrt().repeat(0, N).add(10**-5))
            this.activation = this.weights.repeat(0, N).mul(xhat).add((<Matrix>this.bias).repeat(0, N))
            this.running_mean = this.running_mean.mul(this.momentum).add(mean.mul(1 - this.momentum))
            this.running_var = this.running_var.mul(this.momentum).add(variance.mul(1 - this.momentum))
            this.cache = {variance,diff, xhat}
        }
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        let error
        if(prev_layer.output_error instanceof Matrix) {
            error = <Matrix> (<Matrix>prev_layer.output_error)
        } else {
            error = new Matrix(prev_layer.output_error.toArray())
        }
        let X: Matrix
        if (next_layer instanceof Matrix) {
            X = next_layer
        } else {
            X = <Matrix>(<Layer>next_layer).activation
        }


        const {variance,diff, xhat} = this.cache
        const dout = error.mm((<Matrix>prev_layer.weights).transpose())
        const N = dout.dim().r
        const std_inv = variance.sqrt().inv_el(10**-8)
        const dX_norm = dout.mul(this.weights.repeat(0, N))
        const dVar = dX_norm.mul(diff).sum(0).mul(-0.5).mul(std_inv.pow(3))
        const dMean = dX_norm.mul(std_inv.mul(-1).repeat(0, N)).sum(0).add(dVar.mul(diff.mul(-2).mean(0)))
        this.output_error = dX_norm.mul(std_inv.repeat(0, N)).add(dVar.repeat(0, N).mul(2).mul(diff).div(N)).add(dMean.div(N).repeat(0, N))
        this.errorWeights = dout.mul(xhat).sum(0)
        this.errorBias = dout.sum(0)
    }*/
}
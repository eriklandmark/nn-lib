import Layer from "./layer";
import Tensor from "../../tensor";
import Matrix from "../../matrix";

export default class FlattenLayer extends Layer {

    constructor() {
        super();
    }

    prevShape: number[] = []

    buildLayer(prevLayerShape: number[]) {
        this.prevShape = prevLayerShape
        this.shape = [prevLayerShape.reduce((acc, val) => acc * val)];
    }

    feedForward(input: Layer, isInTraining: boolean) {
        const matrix = new Matrix((<Tensor[]>input.activation).map((tensor) => tensor.vectorize()))
        this.activation = matrix.transpose()
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        const dout = <Matrix> (<Matrix>prev_layer.output_error).mm(prev_layer.weights.transpose())
        let t: Tensor[] = new Array(dout.dim().r);

        for(let i = 0; i < t.length; i++) {
            t[i] = new Tensor();
            t[i].createEmptyArray(this.prevShape[0], this.prevShape[1], this.prevShape[2])
        }

        let [h, w, d] = this.prevShape
        dout.iterate((n: number, i: number) => {
            let r = Math.floor(i / (w*d))
            let c = Math.floor(i / (d) - (r*w))
            let g = Math.floor(i - (c*d) - (r*w*d))
            t[n].set(r,c,g, dout.get(n, i))
        })

        console.log(t[0].toString())

        this.output_error = t
    }
}
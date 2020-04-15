import Layer from "./layer";
import Tensor from "../../tensor";
import Matrix from "../../matrix";
import {SavedLayer} from "../../model";

export default class FlattenLayer extends Layer {

    type: string = "flatten"
    prevShape: number[] = []

    buildLayer(prevLayerShape: number[]) {
        this.prevShape = prevLayerShape
        this.shape = [prevLayerShape.reduce((acc, val) => acc * val)];
    }

    feedForward(input: Layer, isInTraining: boolean) {
        const matrix = new Matrix((<Tensor[]>input.activation).map((tensor) => tensor.vectorize(true)))
        this.activation = matrix.transpose()
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        const dout = <Matrix> (<Matrix>prev_layer.output_error).mm(prev_layer.weights.transpose())
        let t: Tensor[] = new Array(prev_layer.output_error.dim().r);

        for(let i = 0; i < t.length; i++) {
            t[i] = new Tensor();
            t[i].createEmptyArray(this.prevShape[0], this.prevShape[1], this.prevShape[2])
        }

        let [h, w, d] = this.prevShape
        dout.iterate((n: number, i: number) => {
            const r = Math.floor(i / (w*d))
            const c = Math.floor(i / (d) - (r*w))
            const g = Math.floor(i - (c*d) - (r*w*d))
            t[n].set(r,c,g, dout.get(n, i))
        })

        this.output_error = t
    }

    toSavedModel(): SavedLayer {
        return {
            shape: this.prevShape
        }
    }

    fromSavedModel(data: SavedLayer) {
        this.buildLayer(data.shape)
    }

}
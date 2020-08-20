import Layer from "./layer";
import Tensor from "../tensor";
import {SavedLayer} from "../model";

export default class FlattenLayer extends Layer {

    type: string = "flatten"
    prevShape: number[] = []

    buildLayer(prevLayerShape: number[]) {
        this.prevShape = prevLayerShape
        this.shape = [prevLayerShape.reduce((acc, val) => acc * val)];
    }

    feedForward(input: Layer, isInTraining: boolean) {
        this.activation = new Tensor((<Float64Array[][][]>input.activation.t).map((t) =>
            new Tensor(t).vectorize(true).t))
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor) {
        const dout = prev_layer.output_error.dot((<Tensor>prev_layer.weights).transpose())
        this.output_error = new Tensor([prev_layer.output_error.shape[0], this.prevShape[0], this.prevShape[1], this.prevShape[2]], true);

        let [h, w, d] = this.prevShape
        dout.iterate((n: number, i: number) => {
            const r = Math.floor(i / (w*d))
            const c = Math.floor(i / (d) - (w*r))
            const g = Math.floor(i - (c*d) - (w*r*d))
            this.output_error.t[n][r][c][g] = dout.t[n][i]
        })
    }

    updateLayer() {}
}
import Layer from "./layer";
import Tensor from "../tensor";
export default class FlattenLayer extends Layer {
    constructor() {
        super(...arguments);
        this.type = "flatten";
        this.prevShape = [];
    }
    buildLayer(prevLayerShape) {
        this.prevShape = prevLayerShape;
        this.shape = [prevLayerShape.reduce((acc, val) => acc * val)];
    }
    feedForward(input, isInTraining) {
        this.activation = new Tensor(input.activation.t.map((t) => new Tensor(t).vectorize(true).t));
    }
    backPropagation(prev_layer, next_layer) {
        const dout = prev_layer.output_error.dot(prev_layer.weights.transpose());
        this.output_error = new Tensor([prev_layer.output_error.shape[0], this.prevShape[0], this.prevShape[1], this.prevShape[2]], true);
        let [h, w, d] = this.prevShape;
        dout.iterate((n, i) => {
            const r = Math.floor(i / (w * d));
            const c = Math.floor(i / (d) - (w * r));
            const g = Math.floor(i - (c * d) - (w * r * d));
            this.output_error.t[n][r][c][g] = dout.t[n][i];
        });
    }
    updateLayer() { }
}

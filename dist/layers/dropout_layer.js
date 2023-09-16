import Layer from "./layer";
export default class DropoutLayer extends Layer {
    constructor(rate = 0.2) {
        super();
        this.rate = 0;
        this.type = "dropout";
        this.rate = rate;
    }
    buildLayer(prevLayerShape) {
        this.shape = prevLayerShape;
    }
    feedForward(input, isInTraining) {
        this.activation = input.activation;
        if (isInTraining) {
            this.activation.iterate((pos) => {
                if (Math.random() < this.rate) {
                    this.activation.set(pos, 0);
                }
            }, true);
        }
    }
    backPropagation(prev_layer, next_layer) {
        this.weights = prev_layer.weights;
        this.output_error = prev_layer.output_error;
    }
    updateLayer() { }
    toSavedModel() {
        const data = super.toSavedModel();
        data.layer_specific = {
            rate: this.rate
        };
        return data;
    }
    fromSavedModel(data) {
        super.fromSavedModel(data);
        this.rate = data.layer_specific.rate;
    }
}

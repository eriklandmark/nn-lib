import Layer from "./layer";
import Matrix from "../../matrix";
import Tensor from "../../tensor";
import IActivation from "../activations/activations";

export default class ConvolutionLayer extends Layer {

    filterSize: number[] = []
    filters: Tensor = new Tensor()
    prevLayerShape: number[] = []

    constructor(filterSize: number[], activation: IActivation) {
        super();
        this.filterSize = filterSize
    }

    buildLayer(prevLayerShape: number[]) {
        this.prevLayerShape = prevLayerShape
        this.filters.createEmptyArray(this.filterSize[0], this.filterSize[1], this.filterSize[2])
        this.filters.populateRandom()
    }

    feedForward(input: Layer | Tensor[], isInTraining: boolean) {
        let input_images: Tensor[]
        if (input instanceof Layer) {
            input_images = <Tensor[]> input.activation
        } else {
            input_images = <Tensor[]> input
        }

        const [h, w, ch] = this.prevLayerShape
        const [f_h, f_w, nr_f] = this.filterSize
        const patch_width = w - f_w + 1
        const patch_height = h - f_h + 1
        const nr_patches = ch * nr_f

        let new_images: Tensor[] = []

        for (let t = 0; t < input_images.length; t++) {
            let patch = new Tensor();
            patch.createEmptyArray(patch_height, patch_width, nr_patches)
            for (let d = 0; d < ch; d++) {
                for (let f = 0; f < nr_f; f++) {
                    const b = (d * 3) + f
                    for (let r = 0; r < patch_height; r++) {
                        for (let c = 0; c < patch_width; c++) {
                            let val: number = 0
                            for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                    val += input_images[t].get(r + c_f_h, c + c_f_w, d) * this.filters.get(c_f_h, c_f_w, f)
                                }
                            }
                            patch.set(r,c,b, this.activationFunction.normal(val))
                        }
                    }
                }
            }
            new_images.push(patch)
        }

        this.activation = new_images
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        this.weights = prev_layer.weights;
        this.output_error = prev_layer.output_error;
    }

    updateWeights(l_rate: number) {}
}
import Layer from "./layer";
import Matrix from "../../matrix";
import DenseLayer from "./dense_layer";
import IActivation from "../activations/activations";
import ILoss from "../losses/losses";
import MeanSquaredError from "../losses/mean_squared_error";

export default class OutputLayer extends DenseLayer {

    loss: number = 0;
    layerSize: number = 0;
    lossFunction: ILoss = new MeanSquaredError()

    constructor(layerSize: number, activation: IActivation) {
        super(layerSize, activation)
        this.layerSize = layerSize
    }

    public backPropagationOutputLayer(labels: Matrix, next_layer: Layer) {
        this.loss = <number> labels.mul(-1).mul(this.activation.log()).sum()
        const nextActv = next_layer.activation.transpose()
        const gradient = <Matrix> this.lossFunction.derivative(this.activation, labels)
        this.errorBias = gradient
        this.output_error = gradient

        if (this.useGpu) {
            const errorWeightsKernel = this.gpuInstance.createKernel(Matrix.mmGpu())
                .setOutput([labels.dim().c, nextActv.dim().r]).setConstants({mmLength: labels.dim().r});
            errorWeightsKernel.setLoopMaxIterations(Math.max(this.activation.dim().r, nextActv.dim().c))
            this.errorWeights = new Matrix(errorWeightsKernel(nextActv.toNumberArray(), gradient.toNumberArray()) as number[][])
            errorWeightsKernel.destroy()
        } else {
            this.errorWeights = <Matrix> next_layer.activation.transpose().mm(gradient)
        }
    }
}
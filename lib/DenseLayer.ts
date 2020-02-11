import Matrix from "./matrix";
import Vector from "./vector";

export interface InputShape {
    r: number,
    c: number
}

export default class DenseLayer {
    private weights: Matrix
    private biases: Vector

    constructor(inputSize: number, outputSize: number) {
        this.weights = new Matrix()
        this.biases = new Vector(new Float64Array(outputSize))
        this.weights.createEmptyArray(inputSize, outputSize)
    }

    public populateWeightAndBiases() {


        this.biases.iterate((_, index) => {
            this.biases.set(index, )
        })
    }
}
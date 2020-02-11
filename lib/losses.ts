import Vector from "./vector";

export default class Losses {

    public static defLoss(v: Vector, labels: Vector): number {
        if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";

        let sum: number = 0;
        v.iterate((val: number, index: number) => {
            sum += (val - labels.get(index))**2
        })
        return sum;
    }

    public static defLoss_derivative(v: Vector, labels: Vector) {
        if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";

        let sum: number = 0;
        v.iterate((val: number, index: number) => {
            sum += 2 * (val - labels.get(index))
        })
        return sum;
    }
}
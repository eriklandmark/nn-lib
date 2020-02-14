import Vector from "./vector";
import Matrix from "./matrix";

export default class Losses {

    public static defLoss(v: Vector | Matrix, labels: Vector): Vector | Matrix {
        if (v instanceof Vector) {
            if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";
            return labels.sub(v)
        } else {
            return (new Matrix([labels]).transpose().sub(v));
        }
    }

    public static defLoss_derivative(v: Vector, labels: Vector): Vector {
        if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";
        return v.sub(labels)
    }

    public static CrossEntropy_derivatiove(m: Matrix, labels: Vector): Matrix {
        const out = m.copy();
        if (m.count() != labels.size()) throw "Labels and output vector doesn't match size..";
        let k = 0;
        out.iterate((i: number, j: number) => {
            out.set(i,j, -1*(labels.get(k) * (1/out.get(i,j)) + (1 - labels.get(k)) * (1/(1-out.get(i,j)))))
            k += 1
        });
        return out;
    }
}
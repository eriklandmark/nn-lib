import Vector from "./vector";
import Matrix from "./matrix";

export default class Losses {

    public static squared_error(v: Vector | Matrix, labels: Vector): Vector | Matrix {
        if (v instanceof Vector) {
            if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";
            return labels.sub(v).pow(2).mul(0.5)
        } else {
            return new Matrix([labels]).transpose().sub(v).pow(2);
        }
    }

    public static squared_error_derivative(v: Vector | Matrix, labels: Vector): Vector | Matrix {
        if (v instanceof Vector) {
            if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";
            return v.sub(labels)
        } else {
            return new Matrix([labels]).sub(v).pow(2);
        }
    }

    public static CrossEntropy(v: Vector, labels: Vector): Vector {
        const out = new Vector(v.size());
        if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";
        out.iterate((_: number, i: number) => {
            const a = v.get(i);
            const y = labels.get(i)
            if (a == 1) {
                out.set(i, -Math.log(a))
            } else {
                out.set(i, -1*((y*Math.log(a)) + (1 - y)*Math.log(1 - a)))
            }

        });
        return out;
    }

    public static CrossEntropy_derivative(v: Vector, labels: Vector): Vector {
        const out = new Vector(v.size());
        if (v.size() != labels.size()) throw "Labels and output vector doesn't match size..";
        out.iterate((_: number, i: number) => {
            const a = v.get(i);
            const y = labels.get(i)
            out.set(i, (-y / a) + ((1 - y) / (1 - a)))
        });
        return out;
    }
}
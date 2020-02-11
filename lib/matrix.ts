import Vector from "./vector";

export default class Matrix {
    matrix: Array<Float64Array> = [];

    public get: Function = (i:number, j:number) => {return this.matrix[i][j]};
    public set: Function = (i:number, j:number, n:number) => {
        this.matrix[i][j] = n;
    };

    constructor (defaultValue: Array<Float64Array> = []) {
        this.matrix = defaultValue;
    }

    public createEmptyArray(rows:number, columns: number) {
        for (let i = 0; i < rows; i++) {
            //@ts-ignore
            this.matrix.push(new Float32Array(columns))
        }
    }

    public dim() {
        return {r: this.matrix.length, c: this.matrix[0]? this.matrix[0].length : 0}
    }

    public toString = () : string => {
        if (this.matrix.length == 0) {
            return "size 0x0 []"
        } else {
            return this.matrix.reduce((acc, i) => {
                acc += i.reduce((s, i) => {
                    s += i.toString() + " ";
                    return s;
                }, "    ") + "\n"
                return acc;
            }, `Matrix ${this.dim().r}x${this.dim().c} [\n`) + " ]"
        }

    }

    public mm(b: Matrix | Vector): Matrix | Vector {
        if (b instanceof Vector) {
            const v: Vector = b;

            if (v.size() != this.dim().c) {
                throw "Wrong dimension.."
            }

            const c = new Vector();
            for (let i = 0; i < v.size(); i++) {
                let s: number = 0;
                for (let k = 0; k < v.size(); k++) {
                    s += this.matrix[i][k]*v.get(k)
                }
                c.set(i, s)
            }

            return c;
        } else if (b instanceof Matrix) {
            if (b.dim().r != this.dim().c) {
                throw "Wrong dimension.."
            }

            const m: Matrix = b;
            let c = new Matrix();
            c.createEmptyArray(this.dim().r, m.dim().c)
            c.iterate((i, j) => {
                let s: number = 0;
                for (let k = 0; k < m.dim().r; k++) {
                    s += this.matrix[i][k]*m.get(k,j)
                }
                c.set(i, j, s)
            })
            return c
        }
    }

    public copy() {
        return new Matrix(this.matrix)
    }

    public iterate(func: Function): void {
        for (let i:number = 0; i < this.dim().r; i++) {
            for (let j:number = 0; j < this.dim().c; j++) {
                func(i, j);
            }
        }
    }

    public add(b: number | Matrix): Matrix {
        let m = this.copy();
        if (b instanceof Number) {
            let scalar: number = <number>b;
            this.iterate((i, j) => {m.set(i,j, m.get(i,j) + scalar)});
            return m
        } else if (b instanceof Matrix) {
            if (b.dim() != this.dim()) throw "Not in dimension";
            this.iterate((i, j) => {m.set(i,j, m.get(i,j) + b.get(i,j))});
        }
    }

    public mul(scalar: number): Matrix {
        let m = this.copy();
        this.iterate((i, j) => {m.set(i,j, m.get(i,j) * scalar)});
        return m
    }
    public div(scalar: number): Matrix {
        let m = this.copy();
        this.iterate((i, j) => {m.set(i,j, m.get(i,j) / scalar)});
        return m
    }

    public transpose(): Matrix {
        let m = new Matrix()
        m.createEmptyArray(this.dim().c, this.dim().r);
        this.iterate((i, j) => {m.set(j, i, this.get(i,j))});
        return m;
    }
}
import Vector from "./vector";

export default class Matrix {
    matrix: Array<Float64Array> = [];

    public get: Function = (i:number, j:number) => {return this.matrix[i][j]};
    public set: Function = (i:number, j:number, n:number) => {this.matrix[i][j] = n;};

    constructor (defaultValue: Array<Array<number>> | Array<Float64Array> | Array<Vector> = []) {
        if (defaultValue.length > 0 && defaultValue[0] instanceof Float64Array) {
            this.matrix = <Array<Float64Array>>defaultValue
        } else if (defaultValue.length > 0 && defaultValue[0] instanceof Vector){
            const rows = (<Vector> defaultValue[0]).size()
            const cols = defaultValue.length;
            this.createEmptyArray(rows, cols);
            this.iterate((i, j) => {this.set(i,j, (<Vector>defaultValue[j]).get(i))})
        } else {
            for (let i = 0; i < defaultValue.length; i++) {
                this.matrix.push(Float64Array.from(<Array<number>>defaultValue[i]))
            }
        }
    }

    public createEmptyArray(rows:number, columns: number) {
        for (let i = 0; i < rows; i++) {
            this.matrix.push(new Float64Array(columns).fill(0))
        }
    }

    public dim() {
        return {r: this.matrix.length, c: this.matrix[0]? this.matrix[0].length : 0}
    }

    public toString = () : string => {
        if (this.matrix.length == 0) {
            return "Matrix: 0x0 []"
        } else {
            let maxCharCount = 0;
            this.iterate((i, j) => {
                let val = this.get(i,j).toString()
                if(val.length > maxCharCount) maxCharCount = val.length
            })
            maxCharCount = Math.min(maxCharCount, 5)
            return this.matrix.reduce((acc, i) => {
                acc += i.reduce((s, i) => {
                    s += " ".repeat(Math.max(maxCharCount - i.toString().length, 0))
                    s += i.toString().substr(0, maxCharCount) + " ";
                    return s;
                }, "    ") + "\n"
                return acc;
            }, `Matrix: ${this.dim().r}x${this.dim().c} [\n`) + " ]"
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

    public populateRandom() {
        this.iterate((i ,j) => {
            this.set(i, j, (Math.random() - 0.5) * 0.1)
        })
    }

    public mm(b: Matrix | Vector): Matrix | Vector {
        if (b instanceof Vector) {
            const v: Vector = b;

            if (v.size() != this.dim().c) {
                throw "Matrix Multiplication: Wrong dimension.."
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
                throw "Matrix Multiplication: Wrong dimension.."
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

    public add(b: number | Matrix): Matrix {
        let m = this.copy();
        if (typeof b == "number") {
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
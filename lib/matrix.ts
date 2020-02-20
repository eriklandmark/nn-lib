import Vector from "./vector";

export default class Matrix {
    matrix: Array<Float64Array> = [];

    public get: Function = (i: number, j: number) => {
        return this.matrix[i][j]
    };
    public set: Function = (i: number, j: number, n: number) => {
        this.matrix[i][j] = n;
    };
    public count: Function = (i: number, j: number, n: number) => {
        return this.dim().c * this.dim().r;
    };

    constructor(defaultValue: Array<Array<number>> | Array<Float64Array> | Array<Vector> = []) {
        if (defaultValue.length > 0 && defaultValue[0] instanceof Float64Array) {
            this.matrix = <Array<Float64Array>>defaultValue
        } else if (defaultValue.length > 0 && defaultValue[0] instanceof Vector) {
            const rows = (<Vector>defaultValue[0]).size()
            const cols = defaultValue.length;
            this.createEmptyArray(rows, cols);
            this.iterate((i, j) => {
                this.set(i, j, (<Vector>defaultValue[j]).get(i))
            })
        } else {
            for (let i = 0; i < defaultValue.length; i++) {
                this.matrix.push(Float64Array.from(<Array<number>>defaultValue[i]))
            }
        }
    }

    public createEmptyArray(rows: number, columns: number) {
        for (let i = 0; i < rows; i++) {
            this.matrix.push(new Float64Array(columns).fill(0))
        }
    }

    public dim() {
        return {r: this.matrix.length, c: this.matrix[0] ? this.matrix[0].length : 0}
    }

    public toString = (): string => {
        if (this.matrix.length == 0) {
            return "Matrix: 0x0 []"
        } else {
            let maxCharCount = 0;
            this.iterate((i, j) => {
                let val = this.get(i, j).toString()
                if (val.length > maxCharCount) maxCharCount = val.length
            })
            maxCharCount = Math.min(maxCharCount, 7)
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

    public static fromJsonObject(obj: Array<Object>) {
        let m = new Matrix()
        m.createEmptyArray(obj.length, Object.keys(obj[0]).length)
        m.iterate((i,j) => {
            m.set(i, j, obj[i][j])
        })
        return m;
    }

    public copy() {
        let m = new Matrix()
        m.createEmptyArray(this.dim().r, this.dim().c)
        m.iterate((i,j) => {m.set(i, j, this.get(i,j))})
        return m
    }

    public iterate(func: Function): void {
        for (let i: number = 0; i < this.dim().r; i++) {
            for (let j: number = 0; j < this.dim().c; j++) {
                func(i, j);
            }
        }
    }

    public populateRandom() {
        this.iterate((i, j) => {
            this.set(i, j, Math.random() * 2 - 1)
        })
    }

    public mm(b: Matrix | Vector): Matrix | Vector {
        if (b instanceof Vector) {
            const v: Vector = b;

            if (v.size() != this.dim().c) {
                throw "Matrix Multiplication: Wrong dimension.."
            }

            const c = new Vector(this.dim().r);
            for (let i = 0; i < this.dim().r; i++) {
                c.set(i, this.matrix[i].reduce((acc: number, val: number, k: number) => acc + (val * v.get(k)),0))
            }
            return c;
        } else if (b instanceof Matrix) {
            if (b.dim().r != this.dim().c) {
                throw "Matrix Multiplication: Wrong dimension.."
            }

            const m: Matrix = b;
            let c = new Matrix();
            c.createEmptyArray(this.dim().r, m.dim().c)
            c.iterate((i: number, j: number) => {
                c.set(i, j, this.matrix[i].reduce((acc: number, val: number, k: number) => acc + (val * m.get(k, j)),0))
            })
            return c
        }
    }

    public add(b: number | Matrix): Matrix {
        let m = this.copy();
        if (typeof b == "number") {
            let scalar: number = <number>b;
            this.iterate((i, j) => {
                m.set(i, j, m.get(i, j) + scalar)
            });
            return m
        } else if (b instanceof Matrix) {
            if (b.dim().r != this.dim().r || b.dim().c != this.dim().c) throw "Matrix Addition: Not the same dimension";
            this.iterate((i, j) => {
                m.set(i, j, m.get(i, j) + b.get(i, j))
            });
            return m;
        }
    }

    public sub(b: number | Matrix): Matrix {
        let m = this.copy();
        if (typeof b == "number") {
            let scalar: number = <number>b;
            this.iterate((i, j) => {
                m.set(i, j, m.get(i, j) - scalar)
            });
            return m
        } else if (b instanceof Matrix) {
            if (b.dim().r != this.dim().r || b.dim().c != this.dim().c) throw "Matrix Addition: Not the same dimension";
            this.iterate((i, j) => {
                m.set(i, j, m.get(i, j) - b.get(i, j))
            });
            return m;
        }
    }

    public mul(scalar: number): Matrix {
        let m = this.copy();
        this.iterate((i, j) => {m.set(i, j, m.get(i, j) * scalar)});
        return m
    }

    public pow(scalar: number): Matrix {
        let m = this.copy();
        this.iterate((i, j) => {m.set(i, j, m.get(i, j) ** scalar)});
        return m
    }

    public div(scalar: number): Matrix {
        let m = this.copy();
        this.iterate((i, j) => {
            m.set(i, j, m.get(i, j) / scalar)
        });
        return m
    }

    public transpose(): Matrix {
        let m = new Matrix()
        m.createEmptyArray(this.dim().c, this.dim().r);
        this.iterate((i, j) => {
            m.set(j, i, this.get(i, j))
        });
        return m;
    }

    public argmax(i: number = -1, row = true) {
        if (row) {
            if (i < 0) {
                return 0;
            } else {
                return this.matrix[i].reduce((acc:number, va:number, ind) => va < this.matrix[i][acc]? ind : acc, 0)
            }
        } else {
            if (i < 0) {
                return 0;
            } else {
                let maxIndex = 0;
                for (let j = 0; j < this.dim().r; j++) {
                    if (Math.abs(this.get(j,i)) > Math.abs(this.get(maxIndex, i))) {
                        maxIndex = j;
                    }
                }
                return maxIndex;
            }
        }
    }

    public inv() {
        if (this.dim().c == 1 && this.dim().c == 1) {
            return new Matrix([[1/this.get(0,0)]])
        } else if (this.dim().c == 2 && this.dim().c) {
            return new Matrix([
                [this.get(1,1), -this.get(0,1)],
                [-this.get(1,0), this.get(0,0)]
            ]).mul(1/((this.get(0,0)*this.get(1,1)) - (this.get(0,1)*this.get(1,0))))
        }
    }
}
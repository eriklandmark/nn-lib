import Vector from "./vector";
import {StaticPool} from "node-worker-threads-pool";

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

    public toString = (max_rows: number = 10): string => {
        if (this.matrix.length == 0) {
            return "Matrix: 0x0 []"
        } else {
            let maxCharCount = 0;
            this.iterate((i, j) => {
                let val = this.get(i, j).toString()
                if (val.length > maxCharCount) maxCharCount = val.length
            })
            maxCharCount = Math.min(maxCharCount, 7)
            return this.matrix.slice(0, Math.min(max_rows, this.matrix.length)).reduce((acc, i) => {
                acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s, i) => {
                    s += " ".repeat(Math.max(maxCharCount - i.toString().length, 0))
                    s += i.toString().substr(0, maxCharCount) + " ";
                    return s;
                }, "    ")
                acc += i.length > max_rows ? "  ... +" + (i.length - max_rows) + " elements\n" : "\n"
                return acc;
            }, `Matrix: ${this.dim().r}x${this.dim().c} [\n`) + (this.matrix.length > max_rows ?
                "    ... +" + (this.matrix.length - max_rows) + " rows \n]" : " ]")
        }
    }

    public static fromJsonObject(obj: Array<Object>) {
        let m = new Matrix()
        m.createEmptyArray(obj.length, Object.keys(obj[0]).length)
        m.iterate((i, j) => {
            m.set(i, j, obj[i][j])
        })
        return m;
    }

    public copy() {
        let m = new Matrix()
        m.createEmptyArray(this.dim().r, this.dim().c)
        m.iterate((i, j) => {
            m.set(i, j, this.get(i, j))
        })
        return m
    }

    public iterate(func: Function): void {
        for (let i: number = 0; i < this.dim().r; i++) {
            for (let j: number = 0; j < this.dim().c; j++) {
                func(i, j);
            }
        }
    }

    public where(scalar: number): number[] {
        this.iterate((i, j) => {
            if (this.get(i, j) == scalar) {
                return [i, j]
            }
        })
        return [-1, -1]
    }


    public populateRandom() {
        this.iterate((i, j) => {
            this.set(i, j, Math.random() * 2 - 1)
        })
    }

    public empty(): boolean {
        return this.dim().c == 0 || this.dim().r == 0
    }

    public async mm(b: Matrix | Vector): Promise<Matrix | Vector> {
        return new Promise<Matrix | Vector>(async (resolve, reject) => {
            if (b instanceof Vector) {
                if (b.size() != this.dim().c) {
                    reject("Matrix Multiplication (Vector): Wrong dimension..")
                }

                const c = new Vector(this.dim().r);
                for (let i = 0; i < this.dim().r; i++) {
                    c.set(i, this.matrix[i].reduce((acc: number, val: number, k: number) => acc + (val * b.get(k)), 0))
                }
                resolve(c);
            } else if (b instanceof Matrix) {
                if (b.dim().r != this.dim().c)
                    reject("Matrix Multiplication (Matrix): Wrong dimension..")

                let c = new Matrix();
                c.createEmptyArray(this.dim().r, b.dim().c)
                let calculations: Promise<any>[] = []
                const filePath = "./lib/mm_worker.js";

                const pool = new StaticPool({
                    size: 1,
                    task: filePath,
                    workerData: {matrix: this.matrix, bMatrix: b.matrix}
                });

                /*for (let i = 0; i < 20; i++) {
                    (async () => {
                        const num = 40 + Math.trunc(10 * Math.random());

                        // This will choose one idle worker in the pool
                        // to execute your heavy task without blocking
                        // the main thread!
                        const res = await pool.exec(num);

                        console.log(`Fibonacci(${num}) result:`, res);
                    })();
                }*/
                c.iterate((i: number, j: number) => {
                    calculations.push(pool.exec({i:i, j:j}))
                })
                console.log("hee")
                const vals = await Promise.all(calculations)
                for (let {i, j, v} of vals) {
                    console.log("d")
                    c.set(i, j, v)
                }

                console.log("hej")

                resolve(c)
            }
        })
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
            if (b.dim().r != m.dim().r || b.dim().c != m.dim().c) throw "Matrix Addition: Not the same dimension";
            this.iterate((i: number, j: number) => {
                m.set(i, j, m.get(i, j) - b.get(i, j))
            });
            return m;
        }
    }

    public mul(b: number | Matrix): Matrix {
        let m = this.copy();
        if (b instanceof Matrix) {
            if (b.dim().r != m.dim().r || b.dim().c != m.dim().c) throw "Matrix mult: Not the same dimension";
            this.iterate((i, j) => {
                m.set(i, j, m.get(i, j) * b.get(i, j))
            });
        } else {
            this.iterate((i, j) => {
                m.set(i, j, m.get(i, j) * b)
            });
        }

        return m
    }

    public pow(scalar: number): Matrix {
        let m = this.copy();
        this.iterate((i, j) => {
            m.set(i, j, m.get(i, j) ** scalar)
        });
        return m
    }

    public exp(): Matrix {
        let m = this.copy();
        this.iterate((i, j) => {
            m.set(i, j, Math.exp(m.get(i, j)))
        });
        return m
    }

    public log(): Matrix {
        let m = this.copy();
        this.iterate((i, j) => {
            m.set(i, j, Math.log(m.get(i, j)))
        });
        return m
    }

    public sum(axis: number = 0, keepDims = false): number | Matrix {
        if (keepDims) {
            let m = this.copy();
            if (axis == 1) {
                m.matrix.forEach((arr, i) => {
                    const sum = arr.reduce((acc, val) => acc + val, 0);
                    arr.forEach((val, j) => m.set(i, j, sum))
                });
            } else if (axis == 0) {
                const sum = m.matrix.reduce((acc, val) => {
                    acc += val.reduce((acc, val) => acc + val, 0)
                    return acc;
                }, 0);
                this.iterate((i, j) => {
                    m.set(i, j, sum)
                });
                return m;
            } else if (axis == 2) {
                return this.copy()
            }
            return m
        } else {
            if (axis == 0) {
                return this.matrix.reduce((acc, val) => {
                    acc += val.reduce((acc, val) => acc + val, 0)
                    return acc;
                }, 0);
            } else if (axis == 1) {
                let m = new Matrix()
                m.createEmptyArray(this.dim().r, 1)
                this.matrix.forEach((arr, i) => {
                    const sum = arr.reduce((acc, val) => acc + val, 0);
                    m.set(i, 0, sum)
                });
                return m;
            } else if (axis == 2) {
                return this.copy()
            }
        }
    }

    public div(scalar: number | Matrix): Matrix {
        let m = this.copy();
        if (scalar instanceof Matrix) {
            this.iterate((i, j) => {
                m.set(i, j, m.get(i, j) / scalar.get(i, j))
            });
        } else {
            this.iterate((i, j) => {
                m.set(i, j, m.get(i, j) / scalar)
            });
        }
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
                return this.matrix[i].reduce((acc: number, va: number, ind) => va > this.get(i, acc) ? ind : acc, 0)
            }
        } else {
            if (i < 0) {
                return 0;
            } else {
                let maxIndex = 0;
                for (let j = 0; j < this.dim().r; j++) {
                    if (Math.abs(this.get(j, i)) > Math.abs(this.get(maxIndex, i))) {
                        maxIndex = j;
                    }
                }
                return maxIndex;
            }
        }
    }

    public inv() {
        if (this.dim().c == 1 && this.dim().c == 1) {
            return new Matrix([[1 / this.get(0, 0)]])
        } else if (this.dim().c == 2 && this.dim().c) {
            return new Matrix([
                [this.get(1, 1), -this.get(0, 1)],
                [-this.get(1, 0), this.get(0, 0)]
            ]).mul(1 / ((this.get(0, 0) * this.get(1, 1)) - (this.get(0, 1) * this.get(1, 0))))
        }
    }
}
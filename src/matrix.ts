import Vector from "./vector";
//import {StaticPool} from "node-worker-threads-pool";
import {KernelFunction} from "gpu.js";

export default class Matrix {
    matrix: Array<Float32Array> = [];

    public get: Function = (i: number, j: number) => {
        if (!isFinite(this.matrix[i][j])) {
            console.trace()
            throw "Getting Number " + this.matrix[i][j] + " is not finite... \n" +
            " Info: [" + i + "][" + j + "]"
        }
        return this.matrix[i][j]
    };
    public set: Function = (i: number, j: number, n: number) => {
        if (!isFinite(n)) {
            console.trace()
            throw "Number "+ n + " is not Finite..."
        }
        this.matrix[i][j] = n;
    };
    public count: Function = () => {
        return this.dim().c * this.dim().r;
    };

    constructor(defaultValue: Array<Array<number>> | Array<Float32Array> | Array<Vector> = []) {
        if (defaultValue.length > 0 && defaultValue[0] instanceof Float32Array) {
            this.matrix = <Array<Float32Array>>defaultValue
        } else if (defaultValue.length > 0 && defaultValue[0] instanceof Vector) {
            const rows = (<Vector>defaultValue[0]).size()
            const cols = defaultValue.length;
            this.createEmptyArray(rows, cols);
            this.iterate((i: number, j: number) => {
                this.set(i, j, (<Vector>defaultValue[j]).get(i))
            })
        } else {
            for (let i = 0; i < defaultValue.length; i++) {
                this.matrix.push(Float32Array.from(<Array<number>>defaultValue[i]))
            }
        }
    }

    public createEmptyArray(rows: number, columns: number) {
        for (let i = 0; i < rows; i++) {
            this.matrix.push(new Float32Array(columns).fill(0))
        }
    }

    public dim() {
        return {r: this.matrix.length, c: this.matrix[0] ? this.matrix[0].length : 0}
    }

    public toString = (max_rows: number = 10, precision: number = 3): string => {
        if (this.matrix.length == 0) {
            return "Matrix: 0x0 []"
        } else {
            return this.matrix.slice(0, Math.min(max_rows, this.matrix.length)).reduce((acc, i) => {
                acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s, i) => {
                    s += " "//.repeat(Math.max(maxCharCount - i.toPrecision(precision).length, 1))
                    s += this.numberToString(i, precision, true);
                    return s;
                }, "    ")
                acc += i.length > max_rows ? "  ... +" + (i.length - max_rows) + " elements\n" : "\n"
                return acc;
            }, `Matrix: ${this.dim().r}x${this.dim().c} [\n`) + (this.matrix.length > max_rows ?
                "    ... +" + (this.matrix.length - max_rows) + " rows \n]" : " ]")
        }
    }

    private numberToString(nr: number, precision: number = 5, autoFill: boolean = false) {
        const expStr = nr.toExponential()
        return (+expStr.substr(0, expStr.lastIndexOf("e"))).toPrecision(precision)
            + expStr.substr(expStr.lastIndexOf("e")) +
            (autoFill? " ".repeat(4-expStr.substr(expStr.lastIndexOf("e")).length) : "")
    }

    public static fromJsonObject(obj: any[]) {
        return new Matrix(obj.map((row: any) => {
            return Object.keys(row).map((item, index) => row[index.toString()])
        }))
    }

    public toNumberArray(): number[][] {
        return this.matrix.map((floatArray) => [].slice.call(floatArray))
    }

    public copy(full: boolean = true) {
        let m = new Matrix()
        m.createEmptyArray(this.dim().r, this.dim().c)
        if (full) {
            m.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j))
            })
        }
        return m
    }

    public fill(scalar: number): Matrix {
        const m = this.copy(false)
        for (let i = 0; i < this.dim().r; i++) {
            m.matrix[i] = new Float32Array(this.dim().c).fill(scalar)
        }
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
        this.iterate((i: number, j: number) => {
            if (this.get(i, j) == scalar) {
                return [i, j]
            }
        })
        return [-1, -1]
    }

    public populateRandom() {
        this.iterate((i: number, j: number) => {
            this.set(i, j, Math.random() * 2 - 1)
        })
    }

    public empty(): boolean {
        return this.dim().c == 0 || this.dim().r == 0
    }

    public isNaN(): boolean {
        for (let i: number = 0; i < this.dim().r; i++) {
            for (let j: number = 0; j < this.dim().c; j++) {
                if (isNaN(this.matrix[i][j]) || this.matrix[i][j] != this.matrix[i][j] ||
                    !isFinite(this.matrix[i][j])) {
                    return true
                }
            }
        }
        return false
    }

    public repeat(axis = 0, times = 1) {
        if(axis == 0) {
            const m = new Matrix()
            m.createEmptyArray(times, this.dim().c)
            m.matrix.fill(this.matrix[0])
            return m
        }
    }

    public static addGpu(): KernelFunction {
        return function add(a, b) {
            //@ts-ignore
            return a + b;
        }
    }

    public static subGpu(): KernelFunction {
        return function sub(a, b) {
            //@ts-ignore
            return a - b;
        }
    }

    public static multiplyGpu(): KernelFunction {
        return function multiply(a, b) {
            //@ts-ignore
            return a * b;
        }
    }

    public static mmGpu(): KernelFunction {
        return function mm(a, b) {
            let sum = 0;
            //@ts-ignore
            for (let i = 0; i < a[0].length; i++) {
                //@ts-ignore
                sum += a[this.thread.y][i] * b[i][this.thread.x];
            }
            return sum;
        }
    }

    public mm(b: Matrix | Vector): Matrix | Vector {
        if (b instanceof Vector) {
            const v: Vector = b;

            if (v.size() != this.dim().c) {
                console.trace()
                throw "Matrix Multiplication (Vector): Wrong dimension..\n"+
                        "This: [ " + this.dim().r + " , " + this.dim().c + " ] | Other: [ " + b.size() + " ]"
            }

            const c = new Vector(this.dim().r);
            for (let i = 0; i < this.dim().r; i++) {
                c.set(i, this.matrix[i].reduce((acc: number, val: number, k: number) => acc + (val * v.get(k)), 0))
            }
            return c;
        } else {
            if (b.dim().r != this.dim().c) {
                console.trace()
                throw "Matrix Multiplication (Matrix): Wrong dimension..\n"+
                "This: [ " + this.dim().r + " , " + this.dim().c + " ] | Other: [ " + b.dim().r + " , " + b.dim().c + " ]"
            }

            const m: Matrix = b;
            let c = new Matrix();
            c.createEmptyArray(this.dim().r, m.dim().c)
            c.iterate((i: number, j: number) => {
                c.set(i, j, this.matrix[i].reduce((acc: number, val: number, k: number) => acc + (val * m.get(k, j)), 0))
            })
            return c
        }
    }
    /*
    public async mmAsync(b: Matrix | Vector): Promise<Matrix | Vector> {
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

                const pool = new StaticPool({
                    size: Math.min(c.dim().r, 5),
                    task: function (row: any) {
                        const {matrix, bMatrix} = this.workerData
                        let result = (new Float32Array(bMatrix[0].length)).map((_, col) => {
                            return matrix[row].reduce((acc: number, val: number, k: number) => acc + (val * bMatrix[k][col]), 0);
                        })

                        return {i: row, v: result}
                    },
                    workerData: {matrix: this.matrix, bMatrix: b.matrix}
                });

                await (async () => {
                    for (let row = 0; row < c.dim().r; row++) {
                        const {i, v} = await pool.exec(row)
                        for (let col = 0; col < v.length; col++) {
                            c.set(i, col, v[col])
                        }
                    }
                })()

                resolve(c)
            }
        })
    }*/

    public add(b: number | Matrix): Matrix {
        let m = this.copy(false);
        if (b instanceof Matrix) {
            if (b.dim().r != this.dim().r || b.dim().c != this.dim().c) {
                console.trace()
                throw "Matrix Addition: Not the same dimension"
            }
            this.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j) + b.get(i, j))
            });
            return m;
        } else {
            let scalar: number = <number>b;
            this.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j) + scalar)
            });
            return m
        }
    }

    public sub(b: number | Matrix): Matrix {
        let m = this.copy(false);
        if (b instanceof Matrix) {
            if (b.dim().r != m.dim().r || b.dim().c != m.dim().c) {
                console.trace();
                throw "Matrix Subtraction: Not the same dimension";
            }
            this.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j) - b.get(i, j))
            });
            return m;
        } else {
            let scalar: number = <number>b;
            this.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j) - scalar)
            });
            return m
        }
    }

    public mul(b: number | Matrix): Matrix {
        let m = this.copy(false);
        if (b instanceof Matrix) {
            if (b.dim().r != m.dim().r || b.dim().c != m.dim().c) {
                console.trace()
                throw "Matrix mult: Not the same dimension";
            }
            this.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j) * b.get(i, j))
            });
        } else {
            this.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j) * b)
            });
        }

        return m
    }

    public pow(scalar: number): Matrix {
        let m = this.copy(false);
        this.iterate((i: number, j: number) => {
            m.set(i, j, this.get(i, j) ** scalar)
        });
        return m
    }

    public sqrt(): Matrix {
        let m = this.copy(false);
        this.iterate((i: number, j: number) => {
            m.set(i, j, Math.sqrt(this.get(i, j)))
        });
        return m
    }

    public inv_el(eps = 10**-7): Matrix {
        let m = this.copy(false);
        this.iterate((i: number, j: number) => {
            m.set(i, j, this.get(i,j) == 0? 1/(this.get(i,j) + eps) : 1/this.get(i, j))
        });
        return m
    }

    public exp(): Matrix {
        let m = this.copy(false);
        this.iterate((i: number, j: number) => {
            m.set(i, j, Math.exp(this.get(i, j)))
        });
        return m
    }

    public log(): Matrix {
        let m = this.copy();
        this.iterate((i: number, j: number) => {
            m.set(i, j, Math.log(m.get(i, j)))
        });
        return m
    }

    public sum(axis: number = -1, keepDims = false): number | Matrix {
        if (keepDims) {
            let m = this.copy();
            if (axis == 1) {
                m.matrix.forEach((arr, i) => {
                    const sum = arr.reduce((acc, val) => acc + val, 0);
                    arr.forEach((val, j) => m.set(i, j, sum))
                });
            } else if (axis == 0) {
                for (let j = 0; j < this.dim().c; j++) {
                    let sum = 0;
                    for (let i = 0; i < this.dim().r; i++) {
                        sum += this.get(i,j)
                    }
                    for (let i = 0; i < this.dim().r; i++) {
                        m.set(i,j, sum)
                    }
                }
            } else if (axis == -1) {
                const sum = m.matrix.reduce((acc, val) => {
                    acc += val.reduce((acc, val) => acc + val, 0)
                    return acc;
                }, 0);
                this.iterate((i: number, j: number) => {
                    m.set(i, j, sum)
                });
            } else if (axis == 2) {
                return this.copy()
            }
            return m
        } else {
            if (axis == -1) {
                return this.matrix.reduce((acc, val) => {
                    acc += val.reduce((acc, val) => acc + val, 0)
                    return acc;
                }, 0);
            } else if (axis == 0) {
                let m = new Matrix()
                m.createEmptyArray(1, this.dim().c)
                this.iterate((i: number, j: number) => {
                    m.set(0, j, this.get(i, j) + m.get(0, j))
                })
                return m;
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
            return 0
        }
    }

    public div(scalar: number | Matrix): Matrix {
        let m = this.copy(false);
        if (scalar instanceof Matrix) {
            this.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j) / scalar.get(i, j))
            });
        } else {
            this.iterate((i: number, j: number) => {
                m.set(i, j, this.get(i, j) / scalar)
            });
        }
        return m
    }

    public transpose(): Matrix {
        let m = new Matrix()
        m.createEmptyArray(this.dim().c, this.dim().r);
        this.iterate((i: number, j: number) => {
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

    public rowVectors() {
        return this.matrix.map((row) => new Vector(row))
    }

    public mean(axis = -1, keep_dims = false): number | Matrix {
        if (axis == -1) {
            return <number> this.sum(-1, false) / this.count()
        } else if (axis == 0 || axis == 1) {
            return (<Matrix> this.sum(axis, keep_dims)).div(axis == 0? this.dim().r: this.dim().c)
        }
    }
}
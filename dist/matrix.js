"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vector_1 = __importDefault(require("./vector"));
class Matrix {
    constructor(defaultValue = []) {
        this.matrix = [];
        this.get = (i, j) => {
            if (!isFinite(this.matrix[i][j])) {
                console.trace();
                throw "Getting Number " + this.matrix[i][j] + " is not finite... \n" +
                    " Info: [" + i + "][" + j + "]";
            }
            return this.matrix[i][j];
        };
        this.set = (i, j, n) => {
            if (!isFinite(n)) {
                console.trace();
                throw "Number " + n + " is not Finite...";
            }
            this.matrix[i][j] = n;
        };
        this.count = () => {
            return this.dim().c * this.dim().r;
        };
        this.toString = (max_rows = 10, precision = 3) => {
            if (this.matrix.length == 0) {
                return "Matrix: 0x0 []";
            }
            else {
                return this.matrix.slice(0, Math.min(max_rows, this.matrix.length)).reduce((acc, i) => {
                    acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s, i) => {
                        s += " "; //.repeat(Math.max(maxCharCount - i.toPrecision(precision).length, 1))
                        s += this.numberToString(i, precision, true);
                        return s;
                    }, "    ");
                    acc += i.length > max_rows ? "  ... +" + (i.length - max_rows) + " elements\n" : "\n";
                    return acc;
                }, `Matrix: ${this.dim().r}x${this.dim().c} [\n`) + (this.matrix.length > max_rows ?
                    "    ... +" + (this.matrix.length - max_rows) + " rows \n]" : " ]");
            }
        };
        if (defaultValue.length > 0 && defaultValue[0] instanceof Float32Array) {
            this.matrix = defaultValue;
        }
        else if (defaultValue.length > 0 && defaultValue[0] instanceof vector_1.default) {
            const rows = defaultValue[0].size();
            const cols = defaultValue.length;
            this.createEmptyArray(rows, cols);
            this.iterate((i, j) => {
                this.set(i, j, defaultValue[j].get(i));
            });
        }
        else {
            for (let i = 0; i < defaultValue.length; i++) {
                this.matrix.push(Float32Array.from(defaultValue[i]));
            }
        }
    }
    createEmptyArray(rows, columns) {
        for (let i = 0; i < rows; i++) {
            this.matrix.push(new Float32Array(columns).fill(0));
        }
    }
    dim() {
        return { r: this.matrix.length, c: this.matrix[0] ? this.matrix[0].length : 0 };
    }
    numberToString(nr, precision = 5, autoFill = false) {
        const expStr = nr.toExponential();
        return (+expStr.substr(0, expStr.lastIndexOf("e"))).toPrecision(precision)
            + expStr.substr(expStr.lastIndexOf("e")) +
            (autoFill ? " ".repeat(4 - expStr.substr(expStr.lastIndexOf("e")).length) : "");
    }
    static fromJsonObject(obj) {
        return new Matrix(obj.map((row) => {
            return Object.keys(row).map((item, index) => row[index.toString()]);
        }));
    }
    toNumberArray() {
        return this.matrix.map((floatArray) => [].slice.call(floatArray));
    }
    copy(full = true) {
        let m = new Matrix();
        m.createEmptyArray(this.dim().r, this.dim().c);
        if (full) {
            m.iterate((i, j) => {
                m.set(i, j, this.get(i, j));
            });
        }
        return m;
    }
    fill(scalar) {
        const m = this.copy(false);
        for (let i = 0; i < this.dim().r; i++) {
            m.matrix[i] = new Float32Array(this.dim().c).fill(scalar);
        }
        return m;
    }
    iterate(func) {
        for (let i = 0; i < this.dim().r; i++) {
            for (let j = 0; j < this.dim().c; j++) {
                func(i, j);
            }
        }
    }
    where(scalar) {
        this.iterate((i, j) => {
            if (this.get(i, j) == scalar) {
                return [i, j];
            }
        });
        return [-1, -1];
    }
    populateRandom() {
        this.iterate((i, j) => {
            this.set(i, j, Math.random() * 2 - 1);
        });
    }
    empty() {
        return this.dim().c == 0 || this.dim().r == 0;
    }
    isNaN() {
        for (let i = 0; i < this.dim().r; i++) {
            for (let j = 0; j < this.dim().c; j++) {
                if (isNaN(this.matrix[i][j]) || this.matrix[i][j] != this.matrix[i][j] ||
                    !isFinite(this.matrix[i][j])) {
                    return true;
                }
            }
        }
        return false;
    }
    repeat(axis = 0, times = 1) {
        if (axis == 0) {
            const m = new Matrix();
            m.createEmptyArray(times, this.dim().c);
            m.matrix.fill(this.matrix[0]);
            return m;
        }
    }
    static addGpu() {
        return function add(a, b) {
            //@ts-ignore
            return a + b;
        };
    }
    static subGpu() {
        return function sub(a, b) {
            //@ts-ignore
            return a - b;
        };
    }
    static multiplyGpu() {
        return function multiply(a, b) {
            //@ts-ignore
            return a * b;
        };
    }
    static mmGpu() {
        return function mm(a, b) {
            let sum = 0;
            //@ts-ignore
            for (let i = 0; i < a[0].length; i++) {
                //@ts-ignore
                sum += a[this.thread.y][i] * b[i][this.thread.x];
            }
            return sum;
        };
    }
    mm(b) {
        if (b instanceof vector_1.default) {
            const v = b;
            if (v.size() != this.dim().c) {
                console.trace();
                throw "Matrix Multiplication (Vector): Wrong dimension..\n" +
                    "This: [ " + this.dim().r + " , " + this.dim().c + " ] | Other: [ " + b.size() + " ]";
            }
            const c = new vector_1.default(this.dim().r);
            for (let i = 0; i < this.dim().r; i++) {
                c.set(i, this.matrix[i].reduce((acc, val, k) => acc + (val * v.get(k)), 0));
            }
            return c;
        }
        else {
            if (b.dim().r != this.dim().c) {
                console.trace();
                throw "Matrix Multiplication (Matrix): Wrong dimension..\n" +
                    "This: [ " + this.dim().r + " , " + this.dim().c + " ] | Other: [ " + b.dim().r + " , " + b.dim().c + " ]";
            }
            const m = b;
            let c = new Matrix();
            c.createEmptyArray(this.dim().r, m.dim().c);
            c.iterate((i, j) => {
                c.set(i, j, this.matrix[i].reduce((acc, val, k) => acc + (val * m.get(k, j)), 0));
            });
            return c;
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
    add(b) {
        let m = this.copy(false);
        if (b instanceof Matrix) {
            if (b.dim().r != this.dim().r || b.dim().c != this.dim().c) {
                console.trace();
                throw "Matrix Addition: Not the same dimension";
            }
            this.iterate((i, j) => {
                m.set(i, j, this.get(i, j) + b.get(i, j));
            });
            return m;
        }
        else {
            let scalar = b;
            this.iterate((i, j) => {
                m.set(i, j, this.get(i, j) + scalar);
            });
            return m;
        }
    }
    sub(b) {
        let m = this.copy(false);
        if (b instanceof Matrix) {
            if (b.dim().r != m.dim().r || b.dim().c != m.dim().c) {
                console.trace();
                throw "Matrix Subtraction: Not the same dimension";
            }
            this.iterate((i, j) => {
                m.set(i, j, this.get(i, j) - b.get(i, j));
            });
            return m;
        }
        else {
            let scalar = b;
            this.iterate((i, j) => {
                m.set(i, j, this.get(i, j) - scalar);
            });
            return m;
        }
    }
    mul(b) {
        let m = this.copy(false);
        if (b instanceof Matrix) {
            if (b.dim().r != m.dim().r || b.dim().c != m.dim().c) {
                console.trace();
                throw "Matrix mult: Not the same dimension";
            }
            this.iterate((i, j) => {
                m.set(i, j, this.get(i, j) * b.get(i, j));
            });
        }
        else {
            this.iterate((i, j) => {
                m.set(i, j, this.get(i, j) * b);
            });
        }
        return m;
    }
    pow(scalar) {
        let m = this.copy(false);
        this.iterate((i, j) => {
            m.set(i, j, Math.pow(this.get(i, j), scalar));
        });
        return m;
    }
    sqrt() {
        let m = this.copy(false);
        this.iterate((i, j) => {
            m.set(i, j, Math.sqrt(this.get(i, j)));
        });
        return m;
    }
    inv_el(eps = Math.pow(10, -7)) {
        let m = this.copy(false);
        this.iterate((i, j) => {
            m.set(i, j, this.get(i, j) == 0 ? 1 / (this.get(i, j) + eps) : 1 / this.get(i, j));
        });
        return m;
    }
    exp() {
        let m = this.copy(false);
        this.iterate((i, j) => {
            m.set(i, j, Math.exp(this.get(i, j)));
        });
        return m;
    }
    log() {
        let m = this.copy();
        this.iterate((i, j) => {
            m.set(i, j, Math.log(m.get(i, j)));
        });
        return m;
    }
    sum(axis = -1, keepDims = false) {
        if (keepDims) {
            let m = this.copy();
            if (axis == 1) {
                m.matrix.forEach((arr, i) => {
                    const sum = arr.reduce((acc, val) => acc + val, 0);
                    arr.forEach((val, j) => m.set(i, j, sum));
                });
            }
            else if (axis == 0) {
                for (let j = 0; j < this.dim().c; j++) {
                    let sum = 0;
                    for (let i = 0; i < this.dim().r; i++) {
                        sum += this.get(i, j);
                    }
                    for (let i = 0; i < this.dim().r; i++) {
                        m.set(i, j, sum);
                    }
                }
            }
            else if (axis == -1) {
                const sum = m.matrix.reduce((acc, val) => {
                    acc += val.reduce((acc, val) => acc + val, 0);
                    return acc;
                }, 0);
                this.iterate((i, j) => {
                    m.set(i, j, sum);
                });
            }
            else if (axis == 2) {
                return this.copy();
            }
            return m;
        }
        else {
            if (axis == -1) {
                return this.matrix.reduce((acc, val) => {
                    acc += val.reduce((acc, val) => acc + val, 0);
                    return acc;
                }, 0);
            }
            else if (axis == 0) {
                let m = new Matrix();
                m.createEmptyArray(1, this.dim().c);
                this.iterate((i, j) => {
                    m.set(0, j, this.get(i, j) + m.get(0, j));
                });
                return m;
            }
            else if (axis == 1) {
                let m = new Matrix();
                m.createEmptyArray(this.dim().r, 1);
                this.matrix.forEach((arr, i) => {
                    const sum = arr.reduce((acc, val) => acc + val, 0);
                    m.set(i, 0, sum);
                });
                return m;
            }
            else if (axis == 2) {
                return this.copy();
            }
            return 0;
        }
    }
    div(scalar) {
        let m = this.copy(false);
        if (scalar instanceof Matrix) {
            this.iterate((i, j) => {
                m.set(i, j, this.get(i, j) / scalar.get(i, j));
            });
        }
        else {
            this.iterate((i, j) => {
                m.set(i, j, this.get(i, j) / scalar);
            });
        }
        return m;
    }
    transpose() {
        let m = new Matrix();
        m.createEmptyArray(this.dim().c, this.dim().r);
        this.iterate((i, j) => {
            m.set(j, i, this.get(i, j));
        });
        return m;
    }
    argmax(i = -1, row = true) {
        if (row) {
            if (i < 0) {
                return 0;
            }
            else {
                return this.matrix[i].reduce((acc, va, ind) => va > this.get(i, acc) ? ind : acc, 0);
            }
        }
        else {
            if (i < 0) {
                return 0;
            }
            else {
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
    inv() {
        if (this.dim().c == 1 && this.dim().c == 1) {
            return new Matrix([[1 / this.get(0, 0)]]);
        }
        else if (this.dim().c == 2 && this.dim().c) {
            return new Matrix([
                [this.get(1, 1), -this.get(0, 1)],
                [-this.get(1, 0), this.get(0, 0)]
            ]).mul(1 / ((this.get(0, 0) * this.get(1, 1)) - (this.get(0, 1) * this.get(1, 0))));
        }
    }
    rowVectors() {
        return this.matrix.map((row) => new vector_1.default(row));
    }
    mean(axis = -1, keep_dims = false) {
        if (axis == -1) {
            return this.sum(-1, false) / this.count();
        }
        else if (axis == 0 || axis == 1) {
            return this.sum(axis, keep_dims).div(axis == 0 ? this.dim().r : this.dim().c);
        }
    }
}
exports.default = Matrix;

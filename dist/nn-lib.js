var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (Object.hasOwnProperty.call(mod, k)) result[k] = mod[k];
    result["default"] = mod;
    return result;
};
define("matrix", ["require", "exports", "vector", "node-worker-threads-pool"], function (require, exports, vector_1, node_worker_threads_pool_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    vector_1 = __importDefault(vector_1);
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
        mm(b, gpu = false) {
            if (b instanceof vector_1.default) {
                const v = b;
                if (v.size() != this.dim().c) {
                    console.trace();
                    throw "Matrix Multiplication (Vector): Wrong dimension..";
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
                    throw "Matrix Multiplication (Matrix): Wrong dimension..";
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
        mmAsync(b) {
            return __awaiter(this, void 0, void 0, function* () {
                return new Promise((resolve, reject) => __awaiter(this, void 0, void 0, function* () {
                    if (b instanceof vector_1.default) {
                        if (b.size() != this.dim().c) {
                            reject("Matrix Multiplication (Vector): Wrong dimension..");
                        }
                        const c = new vector_1.default(this.dim().r);
                        for (let i = 0; i < this.dim().r; i++) {
                            c.set(i, this.matrix[i].reduce((acc, val, k) => acc + (val * b.get(k)), 0));
                        }
                        resolve(c);
                    }
                    else if (b instanceof Matrix) {
                        if (b.dim().r != this.dim().c)
                            reject("Matrix Multiplication (Matrix): Wrong dimension..");
                        let c = new Matrix();
                        c.createEmptyArray(this.dim().r, b.dim().c);
                        const pool = new node_worker_threads_pool_1.StaticPool({
                            size: Math.min(c.dim().r, 5),
                            task: function (row) {
                                const { matrix, bMatrix } = this.workerData;
                                let result = (new Float32Array(bMatrix[0].length)).map((_, col) => {
                                    return matrix[row].reduce((acc, val, k) => acc + (val * bMatrix[k][col]), 0);
                                });
                                return { i: row, v: result };
                            },
                            workerData: { matrix: this.matrix, bMatrix: b.matrix }
                        });
                        yield (() => __awaiter(this, void 0, void 0, function* () {
                            for (let row = 0; row < c.dim().r; row++) {
                                const { i, v } = yield pool.exec(row);
                                for (let col = 0; col < v.length; col++) {
                                    c.set(i, col, v[col]);
                                }
                            }
                        }))();
                        resolve(c);
                    }
                }));
            });
        }
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
});
define("tensor", ["require", "exports", "vector", "matrix"], function (require, exports, vector_2, matrix_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    vector_2 = __importDefault(vector_2);
    matrix_1 = __importDefault(matrix_1);
    class Tensor {
        constructor(v = []) {
            this.tensor = [];
            this.get = (i, j, k) => {
                if (isNaN(this.tensor[i][j][k])) {
                    console.trace();
                    throw "Getting an NaN...";
                }
                return this.tensor[i][j][k];
            };
            this.set = (i, j, k, n) => {
                if (isNaN(n)) {
                    console.trace();
                    throw "Number is NaN...";
                }
                this.tensor[i][j][k] = n;
            };
            this.count = () => {
                return this.dim().c * this.dim().r * this.dim().d;
            };
            this.toString = (max_rows = 10) => {
                if (this.tensor.length == 0) {
                    return "Tensor: 0x0x0 []";
                }
                else {
                    let maxCharCount = 0;
                    this.iterate((i, j, k) => {
                        let val = this.get(i, j, k).toString();
                        if (val.length > maxCharCount)
                            maxCharCount = val.length;
                    });
                    maxCharCount = Math.min(maxCharCount, 7);
                    let string = `Tensor: ${this.dim().r}x${this.dim().c}x${this.dim().d} [\n`;
                    for (let d = 0; d < this.dim().d; d++) {
                        string += this.tensor.slice(0, Math.min(max_rows, this.tensor.length)).reduce((acc, i) => {
                            acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s, j) => {
                                s += " ".repeat(Math.max(maxCharCount - j[d].toString().length, 0));
                                s += j[d].toString().substr(0, maxCharCount) + " ";
                                return s;
                            }, "    ");
                            acc += i.length > max_rows ? " ... +" + (i.length - max_rows) + " elements\n" : "\n";
                            return acc;
                        }, "") + (this.tensor.length > max_rows ?
                            "    ... +" + (this.tensor.length - max_rows) + " rows \n" : "\n");
                    }
                    return string + "]";
                }
            };
            if (v.length > 0 && v[0][0] instanceof Float32Array) {
                this.tensor = v;
            }
            else {
                for (let i = 0; i < v.length; i++) {
                    this.tensor.push([]);
                    for (let j = 0; j < v[i].length; j++) {
                        this.tensor[i].push(Float32Array.from(v[i][j]));
                    }
                }
            }
        }
        dim() {
            return {
                r: this.tensor.length,
                c: this.tensor[0] ? this.tensor[0].length : 0,
                d: this.tensor[0][0] ? this.tensor[0][0].length : 0
            };
        }
        shape() {
            return [this.dim().r, this.dim().c, this.dim().d];
        }
        createEmptyArray(rows, columns, depth) {
            this.tensor = [];
            for (let i = 0; i < rows; i++) {
                this.tensor.push([]);
                for (let j = 0; j < columns; j++) {
                    this.tensor[i].push(new Float32Array(depth).fill(0));
                }
            }
        }
        static fromJsonObject(obj) {
            return new Tensor(obj.map((row) => {
                return row.map((col) => {
                    return Object.keys(col).map((item, index) => col[index.toString()]);
                });
            }));
        }
        toNumberArray() {
            return this.tensor.map((array) => array.map((floatArray) => [].slice.call(floatArray)));
        }
        iterate(func, channel_first = false) {
            if (channel_first) {
                for (let k = 0; k < this.dim().d; k++) {
                    for (let i = 0; i < this.dim().r; i++) {
                        for (let j = 0; j < this.dim().c; j++) {
                            func(i, j, k);
                        }
                    }
                }
            }
            else {
                for (let i = 0; i < this.dim().r; i++) {
                    for (let j = 0; j < this.dim().c; j++) {
                        for (let k = 0; k < this.dim().d; k++) {
                            func(i, j, k);
                        }
                    }
                }
            }
        }
        copy(full = true) {
            let t = new Tensor();
            t.createEmptyArray(this.dim().r, this.dim().c, this.dim().d);
            if (full) {
                t.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k));
                });
            }
            return t;
        }
        populateRandom() {
            this.iterate((i, j, k) => {
                this.set(i, j, k, Math.random() * 2 - 1);
            });
        }
        empty() {
            return this.dim().c == 0 || this.dim().r == 0 || this.dim().d == 0;
        }
        vectorize(channel_first = false) {
            const v = new vector_2.default(this.count());
            let index = 0;
            this.iterate((i, j, k) => {
                v.set(index, this.get(i, j, k));
                index += 1;
            }, channel_first);
            return v;
        }
        div(val) {
            let t = this.copy(false);
            if (val instanceof Tensor) {
                if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                    console.trace();
                    throw "Tensor Division: Not the same dimension";
                }
                this.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k) / val.get(i, j, k));
                });
            }
            else {
                this.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k) / val);
                });
            }
            return t;
        }
        mul(val) {
            let t = this.copy(false);
            if (val instanceof Tensor) {
                if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                    console.trace();
                    throw "Tensor Multiplication: Not the same dimension";
                }
                this.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k) * val.get(i, j, k));
                });
            }
            else {
                this.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k) * val);
                });
            }
            return t;
        }
        sub(val) {
            let t = this.copy(false);
            if (val instanceof Tensor) {
                if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                    console.trace();
                    throw "Tensor Subtraction: Not the same dimension";
                }
                this.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k) - val.get(i, j, k));
                });
            }
            else {
                this.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k) - val);
                });
            }
            return t;
        }
        add(val) {
            let t = this.copy(false);
            if (val instanceof Tensor) {
                if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                    console.trace();
                    throw "Tensor Subtraction: Not the same dimension";
                }
                this.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k) + val.get(i, j, k));
                });
            }
            else {
                this.iterate((i, j, k) => {
                    t.set(i, j, k, this.get(i, j, k) + val);
                });
            }
            return t;
        }
        padding(padding_height, padding_width) {
            const t = new Tensor();
            t.createEmptyArray(2 * padding_height + this.dim().r, 2 * padding_width + this.dim().c, this.dim().d);
            for (let i = 0; i < this.dim().r; i++) {
                for (let j = 0; j < this.dim().c; j++) {
                    for (let c = 0; c < this.dim().d; c++) {
                        t.set(i + padding_height, j + padding_width, c, this.get(i, j, c));
                    }
                }
            }
            return t;
        }
        im2patches(patch_height, patch_width, filter_height, filter_width) {
            const cols = [];
            for (let r = 0; r < patch_height; r++) {
                for (let c = 0; c < patch_width; c++) {
                    const v = [];
                    for (let c_f_c = 0; c_f_c < this.dim().d; c_f_c++) {
                        for (let c_f_h = 0; c_f_h < filter_height; c_f_h++) {
                            for (let c_f_w = 0; c_f_w < filter_width; c_f_w++) {
                                v.push(this.get(r + c_f_h, c + c_f_w, c_f_c));
                            }
                        }
                    }
                    cols.push(new vector_2.default(v));
                }
            }
            return new matrix_1.default(cols);
        }
        rotate180() {
            const t = this.copy(false);
            this.iterate((i, j, k) => {
                t.set(this.dim().r - 1 - i, this.dim().c - 1 - j, k, this.get(i, j, k));
            });
            return t;
        }
    }
    exports.default = Tensor;
});
define("vector", ["require", "exports", "tensor"], function (require, exports, tensor_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    tensor_1 = __importDefault(tensor_1);
    class Vector {
        constructor(defaultValue = new Float32Array(0)) {
            this.size = () => {
                return this.vector.length;
            };
            this.get = (i) => {
                return this.vector[i];
            };
            this.set = (i, n) => {
                this.vector[i] = n;
            };
            this.toString = (vertical = true) => {
                if (this.vector.length == 0) {
                    return "Vector: []";
                }
                else {
                    if (vertical) {
                        return this.vector.reduce((acc, i) => {
                            acc += `    ${i}\n`;
                            return acc;
                        }, `Vector: [\n`) + " ]";
                    }
                    else {
                        return this.vector.reduce((acc, v) => {
                            acc += v.toString() + " ";
                            return acc;
                        }, "Vector: [ ") + "]";
                    }
                }
            };
            if (defaultValue instanceof Float32Array) {
                this.vector = defaultValue;
            }
            else if (typeof defaultValue == "number") {
                this.vector = new Float32Array(defaultValue);
            }
            else {
                this.vector = Float32Array.from(defaultValue);
            }
        }
        static fromJsonObj(obj) {
            return new Vector(Object.keys(obj).map((item, index) => {
                return obj[index.toString()];
            }));
        }
        static fromBuffer(buff) {
            let v = new Vector(buff.length);
            for (let i = 0; i < v.size(); i++) {
                v.set(i, buff[i]);
            }
            return v;
        }
        static toCategorical(index, size) {
            const v = new Vector(new Float32Array(size).fill(0));
            v.set(index, 1);
            return v;
        }
        toNumberArray() {
            return [].slice.call(this.vector);
        }
        populateRandom() {
            this.iterate((_, index) => {
                this.set(index, Math.random() * 2 - 1);
            });
        }
        iterate(func) {
            this.vector.forEach((value, index) => {
                func(value, index);
            });
        }
        add(b) {
            let v = new Vector(this.size());
            if (b instanceof Vector) {
                if (b.size() != this.size())
                    throw "Vectors to add aren't the same size..";
                this.iterate((val, i) => {
                    v.set(i, val + b.get(i));
                });
                return v;
            }
            else {
                let scalar = b;
                this.iterate((val, i) => {
                    v.set(i, val + scalar);
                });
                return v;
            }
        }
        sub(b) {
            let v = new Vector(this.size());
            if (b instanceof Vector) {
                if (b.size() != this.size())
                    throw "Vectors to subtract aren't the same size..";
                this.iterate((val, i) => {
                    v.set(i, val - b.get(i));
                });
                return v;
            }
            else {
                let scalar = b;
                this.iterate((val, i) => {
                    v.set(i, val - scalar);
                });
                return v;
            }
        }
        mul(input) {
            let v = new Vector(this.size());
            if (input instanceof Vector) {
                if (input.size() != this.size()) {
                    console.trace();
                    throw "Vectors to multiply aren't the same size..";
                }
                this.iterate((val, i) => {
                    v.set(i, val * input.get(i));
                });
            }
            else {
                this.iterate((val, i) => {
                    v.set(i, val * input);
                });
            }
            return v;
        }
        div(scalar) {
            let v = new Vector(this.size());
            this.iterate((val, i) => {
                v.set(i, val / scalar);
            });
            return v;
        }
        pow(scalar) {
            let v = new Vector(this.size());
            this.iterate((val, i) => {
                v.set(i, Math.pow(val, scalar));
            });
            return v;
        }
        exp() {
            let v = new Vector(this.size());
            this.iterate((val, i) => {
                v.set(i, Math.exp(val));
            });
            return v;
        }
        sum() {
            return this.vector.reduce((acc, val) => acc + val);
        }
        mean() {
            return this.sum() / this.size();
        }
        argmax() {
            return this.vector.reduce((acc, va, ind) => va > this.get(acc) ? ind : acc, 0);
        }
        reshape(shape) {
            if (this.size() != shape.reduce((acc, n) => acc * n, 1)) {
                throw "Product of shape must be the same as size of vector!";
            }
            const t = new tensor_1.default();
            t.createEmptyArray(shape[0], shape[1], shape[2]);
            let [h, w, d] = shape;
            this.iterate((val, i) => {
                const r = Math.floor(i / (w * d));
                const c = Math.floor(i / (d) - (r * w));
                const g = Math.floor(i - (c * d) - (r * w * d));
                t.set(r, c, g, val);
            });
            return t;
        }
        normalize() {
            return this.div(this.size());
        }
    }
    exports.default = Vector;
});
define("helpers/array_helper", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class ArrayHelper {
        static shuffle(array) {
            let currentIndex = array.length, temporaryValue, randomIndex;
            while (0 !== currentIndex) {
                randomIndex = Math.floor(Math.random() * currentIndex);
                currentIndex -= 1;
                temporaryValue = array[currentIndex];
                array[currentIndex] = array[randomIndex];
                array[randomIndex] = temporaryValue;
            }
            return array;
        }
        static flatten(array) {
            const new_array = [];
            const flatten_rec = (a) => {
                for (let item of a) {
                    if (Array.isArray(item)) {
                        flatten_rec(item);
                    }
                    else {
                        new_array.push(item);
                    }
                }
            };
            flatten_rec(array);
            return new_array;
        }
        static delete_doublets(array) {
            return array.reduce((acc, el) => {
                if (!acc.includes(el))
                    acc.push(el);
                return acc;
            }, []);
        }
    }
    exports.default = ArrayHelper;
});
define("dataset", ["require", "exports", "vector", "fs", "path", "jimp", "tensor", "helpers/array_helper"], function (require, exports, vector_3, fs, path, jimp_1, tensor_2, array_helper_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    vector_3 = __importDefault(vector_3);
    fs = __importStar(fs);
    path = __importStar(path);
    jimp_1 = __importDefault(jimp_1);
    tensor_2 = __importDefault(tensor_2);
    array_helper_1 = __importDefault(array_helper_1);
    class Dataset {
        constructor() {
            this.data = [];
            this.BATCH_SIZE = 1;
            this.IS_GENERATOR = false;
            this.TOTAL_EXAMPLES = 0;
            this.DATA_STRUCTURE = undefined;
            this.GENERATOR = () => {
            };
        }
        size() {
            return this.data.length;
        }
        setGenerator(gen) {
            this.GENERATOR = gen;
        }
        static read_image(path) {
            return __awaiter(this, void 0, void 0, function* () {
                const image = yield jimp_1.default.read(path);
                const t = new tensor_2.default();
                for (let i = 0; i < image.bitmap.data.length; i += 4) {
                    let y = Math.floor((i / 4) / image.getHeight());
                    let x = (i / 4) - (y * image.getWidth());
                    for (let d = 0; d < 3; d++) {
                        t.set(y, x, d, image.bitmap.data[i + d]);
                    }
                }
                return t;
            });
        }
        vectorize_image(image) {
            const v = new vector_3.default(image.count());
            let index = 0;
            image.iterate((i, j, k) => {
                v.set(index, image.get(i, j, k));
                index += 1;
            });
            this.DATA_STRUCTURE = vector_3.default;
            return v;
        }
        loadMnistTrain(folderPath, maxExamples = 60000, vectorize = true) {
            this.loadMnist(folderPath, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", maxExamples, vectorize);
        }
        loadMnistTest(folderPath, maxExamples = 60000, vectorize = true) {
            this.loadMnist(folderPath, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", maxExamples, vectorize);
        }
        shuffle() {
            this.data = array_helper_1.default.shuffle(this.data);
        }
        loadMnist(folderPath, imageFileName, labelFileName, maxExamples, vectorize) {
            const trainFileBuffer = fs.readFileSync(path.join(folderPath + "/" + imageFileName));
            const labelFileBuffer = fs.readFileSync(path.join(folderPath + "/" + labelFileName));
            for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
                const image = new tensor_2.default();
                const size = 28;
                image.createEmptyArray(size, size, 1 /*vectorize? 1: 3*/);
                for (let x = 0; x < size; x++) {
                    for (let y = 0; y < size; y++) {
                        const val = trainFileBuffer[(imageIndex * size * size) + (x + (y * size)) + 15];
                        if (isNaN(val)) {
                            console.log("Failes", val);
                        }
                        image.set(y, x, 0, val);
                        /*if (!vectorize) {
                            image.set(y, x, 1, val)
                            image.set(y, x, 2, val)
                        }*/
                    }
                }
                let exampleData;
                if (vectorize) {
                    exampleData = this.vectorize_image(image);
                }
                else {
                    exampleData = image;
                    this.DATA_STRUCTURE = tensor_2.default;
                }
                exampleData = exampleData.div(255);
                let example = {
                    data: exampleData,
                    label: vector_3.default.toCategorical(labelFileBuffer[imageIndex + 8], 10)
                };
                this.data.push(example);
            }
        }
        loadTestData(path, maxExamples = 2100) {
            const data = JSON.parse(fs.readFileSync(path, { encoding: "UTF-8" }));
            for (let imageIndex = 0; imageIndex < maxExamples; imageIndex++) {
                let example = {
                    data: new vector_3.default(data["features"][imageIndex]),
                    label: vector_3.default.toCategorical(data["labels"][imageIndex], 3)
                };
                this.data.push(example);
            }
        }
        getBatch(batch) {
            return this.data.slice(batch * this.BATCH_SIZE, batch * this.BATCH_SIZE + this.BATCH_SIZE);
        }
    }
    exports.default = Dataset;
});
define("main", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.NAME = "nn-lib";
});
define("activations/sigmoid", ["require", "exports", "matrix"], function (require, exports, matrix_2) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    matrix_2 = __importDefault(matrix_2);
    class Sigmoid {
        constructor() {
            this.name = "sigmoid";
        }
        normal_gpu() {
            return function actv(a) {
                return (1 / (1 + Math.exp(-a[this.thread.y][this.thread.x])));
            };
        }
        derivative_gpu() {
            return function actv_der(a) {
                return a * (1 - a);
            };
        }
        normal(input) {
            if (input instanceof matrix_2.default) {
                const m = input.copy(false);
                m.iterate((i, j) => {
                    m.set(i, j, 1 / (1 + Math.exp(-input.get(i, j))));
                });
                return m;
            }
            else {
                return 1 / (1 + Math.exp(-input));
            }
        }
        derivative(input) {
            if (input instanceof matrix_2.default) {
                const m = input.copy(false);
                m.iterate((i, j) => {
                    m.set(i, j, input.get(i, j) * (1 - input.get(i, j)));
                });
                return m;
            }
            else {
                return input * (1 - input);
            }
        }
    }
    exports.default = Sigmoid;
});
define("activations/relu", ["require", "exports", "matrix"], function (require, exports, matrix_3) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    matrix_3 = __importDefault(matrix_3);
    class ReLu {
        constructor() {
            this.name = "relu";
        }
        normal(input) {
            if (input instanceof matrix_3.default) {
                const m = input.copy(false);
                m.iterate((i, j) => {
                    m.set(i, j, Math.max(input.get(i, j), 0));
                });
                return m;
            }
            else {
                return Math.max(input, 0);
            }
        }
        derivative(input) {
            if (input instanceof matrix_3.default) {
                const m = input.copy(false);
                m.iterate((i, j) => {
                    m.set(i, j, input.get(i, j) > 0 ? 1 : 0);
                });
                return m;
            }
            else {
                return input > 0 ? 1 : 0;
            }
        }
        normal_gpu() {
            return function actv(a) {
                return Math.max(a[this.thread.y][this.thread.x], 0);
            };
        }
        derivative_gpu() {
            return function actv(a) {
                return a > 0 ? 1 : 0;
            };
        }
    }
    exports.default = ReLu;
});
define("activations/softmax", ["require", "exports", "matrix"], function (require, exports, matrix_4) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    matrix_4 = __importDefault(matrix_4);
    class Softmax {
        constructor() {
            this.name = "softmax";
        }
        normal(input) {
            const exp = input.exp();
            return exp.div(exp.sum(1, true));
        }
        derivative(input) {
            if (input instanceof matrix_4.default) {
                const m = input.copy(false);
                m.iterate((i, j) => {
                    if (i == j) {
                        m.set(i, j, input.get(i, j) * (1 - input.get(i, j)));
                    }
                    else {
                        m.set(i, j, -(input.get(i, j) * input.get(i, j)));
                    }
                });
                return m;
            }
            else {
                return input * (1 - input);
            }
        }
        normal_gpu() {
            return function actv(a) {
                let sum = 0;
                for (let i = 0; i < this.constants.softmax; i++) {
                    sum += Math.exp(a[this.thread.y][i]);
                }
                return Math.exp(a[this.thread.y][this.thread.x]) / sum;
            };
        }
        derivative_gpu() {
            return function actv_der(a) {
                return a * (1 - a);
            };
        }
    }
    exports.default = Softmax;
});
define("activations/hyperbolic_tangent", ["require", "exports", "matrix"], function (require, exports, matrix_5) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    matrix_5 = __importDefault(matrix_5);
    class HyperbolicTangent {
        constructor() {
            this.name = "tanh";
        }
        normal_gpu() {
            return function actv(a) {
                return Math.tanh(a[this.thread.y][this.thread.x]);
            };
        }
        derivative_gpu() {
            return function actv_der(a) {
                return 1 - Math.pow(a, 2);
            };
        }
        normal(input) {
            if (input instanceof matrix_5.default) {
                const m = input.copy(false);
                m.iterate((i, j) => {
                    m.set(i, j, Math.tanh(input.get(i, j)));
                });
                return m;
            }
            else {
                return Math.tanh(input);
            }
        }
        derivative(input) {
            if (input instanceof matrix_5.default) {
                const m = input.copy(false);
                m.iterate((i, j) => {
                    m.set(i, j, 1 - Math.pow(input.get(i, j), 2));
                });
                return m;
            }
            else {
                return 1 - Math.pow(input, 2);
            }
        }
    }
    exports.default = HyperbolicTangent;
});
define("activations/activations", ["require", "exports", "activations/sigmoid", "activations/relu", "activations/softmax", "activations/hyperbolic_tangent"], function (require, exports, sigmoid_1, relu_1, softmax_1, hyperbolic_tangent_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    sigmoid_1 = __importDefault(sigmoid_1);
    relu_1 = __importDefault(relu_1);
    softmax_1 = __importDefault(softmax_1);
    hyperbolic_tangent_1 = __importDefault(hyperbolic_tangent_1);
    class Activation {
        static fromName(name) {
            switch (name) {
                case "sigmoid": return new sigmoid_1.default();
                case "relu": return new relu_1.default();
                case "softmax": return new softmax_1.default();
                case "tanh": return new hyperbolic_tangent_1.default();
                default: return new sigmoid_1.default();
            }
        }
    }
    exports.default = Activation;
});
define("layers/layer", ["require", "exports", "matrix", "vector", "gpu.js"], function (require, exports, matrix_6, vector_4, gpu_js_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    matrix_6 = __importDefault(matrix_6);
    vector_4 = __importDefault(vector_4);
    class Layer {
        constructor() {
            this.weights = new matrix_6.default();
            this.bias = new vector_4.default();
            this.errorWeights = new matrix_6.default();
            this.errorBias = new matrix_6.default();
            this.activation = new matrix_6.default();
            this.useGpu = false;
            this.gpuInstance = new gpu_js_1.GPU();
            this.shape = [];
            this.prevLayerShape = [];
            this.type = "";
            this.hasGPUSupport = false;
            this.isFirstLayer = false;
        }
        setGpuInstance(gpuIns) {
            this.gpuInstance = gpuIns;
        }
        getLayerInfo() {
            return {
                type: this.type,
                shape: this.shape,
                activation: this.activationFunction ? this.activationFunction.name : "NO ACTIVATION"
            };
        }
        buildLayer(prevLayerShape) { }
        feedForward(input, isInTraining) { }
        buildFFKernels(batch_size) { }
        buildBPKernels(size) { }
        backPropagation(prev_layer, next_layer) { }
        calculate_errors(error, next_layer) { }
        updateWeights(l_rate) { }
        toSavedModel() { return; }
        fromSavedModel(data) { }
    }
    exports.default = Layer;
});
define("losses/cross_entropy", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class CrossEntropy {
        constructor() {
            this.name = "cross_entropy";
            this.epsilon = Math.pow(10, -14);
        }
        normal(input, labels) {
            let out = input.copy(false);
            out.iterate((i, j) => {
                if (labels.get(i, j) != 0) {
                    out.set(i, j, (labels.get(i, j) * Math.log(input.get(i, j) + this.epsilon)));
                }
            });
            return out.sum(1, true).mul(-1);
        }
        derivative(input, labels) {
            return labels.mul(-1).div(input);
        }
        normal_gpu() {
            return function actv(a, labels) {
                let sum = 0;
                for (let i = 0; i < this.constants.labels_length; i++) {
                    sum += labels[this.thread.y][i] * Math.log(a[this.thread.y][i] + Math.pow(10, -14));
                }
                return sum * -1;
            };
        }
        derivative_gpu() {
            return function actv(a, labels) {
                labels;
            };
        }
    }
    exports.default = CrossEntropy;
});
define("losses/mean_squared_error", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class MeanSquaredError {
        constructor() {
            this.name = "mean_squared_error";
        }
        normal(input, labels) {
            if (input.dim().r != labels.dim().r || input.dim().c != labels.dim().c)
                throw "Labels and output vector doesn't match size..";
            return input.sub(labels).pow(2);
        }
        derivative(input, labels) {
            if (input.dim().r != labels.dim().r || input.dim().c != labels.dim().c)
                throw "Labels and output vector doesn't match size..";
            return input.sub(labels);
        }
        normal_gpu() {
            return function actv() {
                return;
            };
        }
        derivative_gpu() {
            return function loss(m, label) {
                //@ts-ignore
                return m - label;
            };
        }
    }
    exports.default = MeanSquaredError;
});
define("losses/losses", ["require", "exports", "vector", "losses/cross_entropy", "losses/mean_squared_error"], function (require, exports, vector_5, cross_entropy_1, mean_squared_error_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    vector_5 = __importDefault(vector_5);
    cross_entropy_1 = __importDefault(cross_entropy_1);
    mean_squared_error_1 = __importDefault(mean_squared_error_1);
    class Losses {
        static fromName(name) {
            switch (name) {
                case "cross_entropy": return new cross_entropy_1.default();
                case "mean_squared_error": return new mean_squared_error_1.default();
            }
        }
        static CrossEntropy(v, labels) {
            const out = new vector_5.default(v.size());
            if (v.size() != labels.size())
                throw "Labels and output vector doesn't match size..";
            out.iterate((_, i) => {
                const a = v.get(i);
                const y = labels.get(i);
                if (a == 1) {
                    out.set(i, -Math.log(a));
                }
                else {
                    out.set(i, -1 * ((y * Math.log(a)) + (1 - y) * Math.log(1 - a)));
                }
            });
            return out;
        }
        static CrossEntropy_derivative(v, labels) {
            const out = new vector_5.default(v.size());
            if (v.size() != labels.size())
                throw "Labels and output vector doesn't match size..";
            out.iterate((_, i) => {
                const a = v.get(i);
                const y = labels.get(i);
                out.set(i, (-y / a) + ((1 - y) / (1 - a)));
            });
            return out;
        }
    }
    exports.default = Losses;
});
define("layers/conv_layer", ["require", "exports", "layers/layer", "tensor", "activations/activations", "vector", "matrix"], function (require, exports, layer_1, tensor_3, activations_1, vector_6, matrix_7) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    layer_1 = __importDefault(layer_1);
    tensor_3 = __importDefault(tensor_3);
    activations_1 = __importDefault(activations_1);
    vector_6 = __importDefault(vector_6);
    matrix_7 = __importDefault(matrix_7);
    class ConvolutionLayer extends layer_1.default {
        constructor(nr_filters = 3, filterSize = [3, 3], ch_first = false, activation) {
            super();
            this.filterSize = [];
            this.filters = [];
            this.padding = 0;
            this.stride = 1;
            this.nr_filters = 0;
            this.errorFilters = [];
            this.errorInput = [];
            this.channel_first = true;
            this.useMM = false;
            this.channel_first = ch_first;
            this.activationFunction = activation;
            this.filterSize = filterSize;
            this.nr_filters = nr_filters;
            this.errorBias = new vector_6.default(nr_filters);
            this.type = "conv";
        }
        buildLayer(prevLayerShape) {
            let h, w;
            const [f_h, f_w] = this.filterSize;
            if (this.channel_first) {
                h = prevLayerShape[1];
                w = prevLayerShape[2];
            }
            else {
                h = prevLayerShape[0];
                w = prevLayerShape[1];
            }
            this.shape = [
                ((h + 2 * this.padding) - f_h + 1) / this.stride,
                ((w + 2 * this.padding) - f_w + 1) / this.stride,
                this.nr_filters
            ];
            this.prevLayerShape = prevLayerShape;
            for (let i = 0; i < this.nr_filters; i++) {
                const filter = new tensor_3.default();
                if (this.channel_first) {
                    filter.createEmptyArray(prevLayerShape[0], this.filterSize[0], this.filterSize[1]);
                }
                else {
                    filter.createEmptyArray(this.filterSize[0], this.filterSize[1], prevLayerShape[2]);
                }
                filter.populateRandom();
                this.filters.push(filter);
            }
            this.bias = new vector_6.default(this.nr_filters);
            this.bias.populateRandom();
        }
        feedForward(input, isInTraining) {
            if (false) {
            }
            else {
                let input_images;
                if (input instanceof layer_1.default) {
                    input_images = input.activation;
                }
                else {
                    input_images = input;
                }
                const ch = this.channel_first ? this.prevLayerShape[0] : this.prevLayerShape[2];
                const [f_h, f_w] = this.filterSize;
                const patch_width = this.shape[1];
                const patch_height = this.shape[0];
                let new_images = [];
                if (this.useMM) {
                    const filterMatrix = new matrix_7.default(this.filters.map((t) => t.vectorize(true))).transpose();
                    for (let t = 0; t < input_images.length; t++) {
                        const convolutionMatrix = filterMatrix.mm(input_images[t].im2patches(patch_height, patch_width, f_h, f_w));
                        const activationMatrix = this.activationFunction.normal(convolutionMatrix);
                        const convTensors = activationMatrix.rowVectors().map((v) => v.reshape([patch_height, patch_width, 1]));
                        const patch = new tensor_3.default();
                        patch.createEmptyArray(patch_height, patch_width, this.nr_filters);
                        for (let i = 0; i < this.nr_filters; i++) {
                            patch.iterate((x, y, _) => {
                                patch.set(x, y, i, convTensors[i].get(x, y, 0));
                            });
                        }
                        new_images.push(patch);
                    }
                }
                else {
                    for (let t = 0; t < input_images.length; t++) {
                        let patch = new tensor_3.default();
                        if (this.channel_first) {
                            patch.createEmptyArray(this.nr_filters, patch_height, patch_width);
                        }
                        else {
                            patch.createEmptyArray(patch_height, patch_width, this.nr_filters);
                        }
                        for (let f = 0; f < this.nr_filters; f++) {
                            for (let r = 0; r < patch_height; r++) {
                                for (let c = 0; c < patch_width; c++) {
                                    let val = 0;
                                    for (let c_f_c = 0; c_f_c < ch; c_f_c++) {
                                        for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                            for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                                if (this.channel_first) {
                                                    val += input_images[t].get(c_f_c, r + c_f_h, c + c_f_w) *
                                                        this.filters[f].get(c_f_h, c_f_w, c_f_c);
                                                }
                                                else {
                                                    val += input_images[t].get(r + c_f_h, c + c_f_w, c_f_c) *
                                                        this.filters[f].get(c_f_h, c_f_w, c_f_c);
                                                }
                                            }
                                        }
                                    }
                                    if (this.channel_first) {
                                        patch.set(f, r, c, this.activationFunction.normal(val) + this.bias.get(f));
                                    }
                                    else {
                                        patch.set(r, c, f, this.activationFunction.normal(val) + this.bias.get(f));
                                    }
                                }
                            }
                        }
                        new_images.push(patch);
                    }
                }
                this.activation = new_images;
            }
        }
        buildFFKernels(batch_size) {
            const output_shape = [this.weights.dim().c, batch_size];
            this.ff_kernel = this.gpuInstance.createKernel(function (image, filter) {
                let val = 0;
                for (let c_f_c = 0; c_f_c < this.constants.channels; c_f_c++) {
                    for (let c_f_h = 0; c_f_h < this.constants.filter_height; c_f_h++) {
                        for (let c_f_w = 0; c_f_w < this.constants.filter_width; c_f_w++) {
                            if (this.constants.channel_first) {
                                val += image[c_f_c][this.thread.y + c_f_h][this.thread.x + c_f_w] * filter[c_f_h][c_f_w][c_f_c];
                            }
                            else {
                                val += image[this.thread.y + c_f_h][this.thread.x + c_f_w][c_f_c] * filter[c_f_h][c_f_w][c_f_c];
                            }
                        }
                    }
                }
                return val;
            }).setConstants({
                channels: this.channel_first ? this.prevLayerShape[0] : this.prevLayerShape[2],
                filter_height: this.filterSize[0],
                filter_width: this.filterSize[1],
                channel_first: this.channel_first
            }).setPrecision("single")
                .setOutput(this.shape);
            this.ff_kernel.immutable = true;
            this.act_kernel = this.gpuInstance.createKernel(this.activationFunction.normal_gpu())
                .setPipeline(true)
                .setConstants({ softmax: this.weights.dim().c })
                .setPrecision("single")
                .setDynamicOutput(false)
                .setOutput(output_shape);
            this.act_kernel.immutable = true;
        }
        backPropagation(prev_layer, next_layer) {
            let input;
            let prev_layer_output = prev_layer.output_error;
            if (next_layer instanceof layer_1.default) {
                input = next_layer.activation;
            }
            else {
                input = next_layer;
            }
            let dout = [];
            prev_layer_output.forEach((t, index) => {
                let ex = t.copy(false);
                ex.iterate((x, y, z) => {
                    const dActv = this.activationFunction.derivative(this.activation[index].get(x, y, z));
                    ex.set(x, y, z, t.get(x, y, z) * dActv);
                });
                dout.push(ex.rotate180());
            });
            const N = input.length;
            const [h, w, ch] = this.prevLayerShape; // X
            const [f_h, f_w] = this.filterSize; // W
            const patch_width = dout[0].dim().c;
            const patch_height = dout[0].dim().r;
            const patch_depth = dout[0].dim().d;
            const padding_width = f_w - 1;
            const padding_height = f_h - 1;
            if (this.useMM) {
                const filterMatrix = new matrix_7.default(dout.map((t) => t.vectorize(true))).transpose();
                this.errorFilters = [];
                for (let t = 0; t < input.length; t++) {
                    const inputMatrix = input[t].im2patches(patch_height, patch_width, f_h, f_w);
                    console.log(filterMatrix.dim(), inputMatrix.dim());
                    const convolutionMatrix = inputMatrix.mm(filterMatrix);
                    const convTensors = convolutionMatrix.rowVectors().map((v) => v.reshape([patch_height, patch_width, 1]));
                    const patch = new tensor_3.default();
                    patch.createEmptyArray(patch_height, patch_width, this.nr_filters);
                    for (let i = 0; i < this.nr_filters; i++) {
                        patch.iterate((x, y, _) => {
                            patch.set(x, y, i, convTensors[i].get(x, y, 0));
                        });
                    }
                    this.errorFilters.push(patch);
                }
                console.log(this.errorFilters);
            }
            else {
                this.errorFilters = this.filters.map((filter) => filter.copy(false));
                this.errorInput = input.map((inp) => inp.copy(false));
                for (let n = 0; n < N; n++) {
                    for (let f = 0; f < this.nr_filters; f++) {
                        for (let i = 0; i < f_h; i++) {
                            for (let j = 0; j < f_w; j++) {
                                for (let k = 0; k < patch_height; k++) {
                                    for (let l = 0; l < patch_width; l++) {
                                        for (let c = 0; c < ch; c++) {
                                            this.errorFilters[f].set(i, j, c, this.errorFilters[f].get(i, j, c) + (input[n].get(this.stride * i + k, this.stride * j + l, c) *
                                                dout[n].get(k, l, f)));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                const sum = [];
                for (let n = 0; n < dout.length; n++) {
                    const sumVector = new vector_6.default(dout[n].dim().d);
                    dout[n].iterate((i, j, k) => {
                        sumVector.set(k, sumVector.get(k) + dout[n].get(i, j, k));
                    });
                    sum.push(sumVector);
                }
                this.errorBias = sum.reduce((acc, v) => acc.add(v), new vector_6.default(sum[0].size())).div(sum.length);
                if (!this.isFirstLayer) {
                    const doutp = new Array(dout.length).fill(new tensor_3.default());
                    doutp.forEach((tensor) => {
                        tensor.createEmptyArray(2 * padding_height + patch_height, 2 * padding_width + patch_width, patch_depth);
                    });
                    for (let n = 0; n < doutp.length; n++) {
                        for (let i = 0; i < patch_height; i++) {
                            for (let j = 0; j < patch_width; j++) {
                                for (let c = 0; c < patch_depth; c++) {
                                    doutp[n].set(i + padding_height, j + padding_width, c, dout[n].get(i, j, c));
                                }
                            }
                        }
                    }
                    const filterInv = this.filters.map((f) => f.copy(false));
                    for (let n = 0; n < filterInv.length; n++) {
                        filterInv[n].iterate((i, j, k) => {
                            filterInv[n].set(filterInv[n].dim().r - 1 - i, filterInv[n].dim().c - 1 - j, k, this.filters[n].get(i, j, k));
                        });
                    }
                    for (let n = 0; n < N; n++) {
                        for (let f = 0; f < this.nr_filters; f++) {
                            for (let i = 0; i < h + (2 * this.padding); i++) {
                                for (let j = 0; j < w + (2 * this.padding); j++) {
                                    for (let k = 0; k < f_h; k++) {
                                        for (let l = 0; l < f_w; l++) {
                                            for (let c = 0; c < ch; c++) {
                                                this.errorInput[n].set(i, j, c, this.errorInput[n].get(i, j, c) + (doutp[n].get(i + k, j + l, f) * filterInv[f].get(k, l, c)));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    this.output_error = this.errorInput;
                }
            }
        }
        convolve(image, filters, channel_first = false) {
            const f_h = filters[0].dim().r;
            const f_w = filters[0].dim().c;
            const patch_width = ((image.dim().r + 2 * this.padding) - f_h + 1) / this.stride;
            const patch_height = ((image.dim().c + 2 * this.padding) - f_w + 1) / this.stride;
            if (this.useMM) {
                const filterMatrix = new matrix_7.default(filters.map((t) => t.vectorize(true))).transpose();
                return filterMatrix.mm(image.im2patches(patch_height, patch_width, filters[0].dim().r, filters[0].dim().c))
                    .rowVectors().map((v) => v.reshape([patch_height, patch_width, 1]));
            }
            else {
                let patch = new tensor_3.default();
                if (channel_first) {
                    patch.createEmptyArray(filters.length, patch_height, patch_width);
                }
                else {
                    patch.createEmptyArray(patch_height, patch_width, filters.length);
                }
                const chs = channel_first ? image.dim().r : image.dim().d;
                for (let f = 0; f < filters.length; f++) {
                    for (let r = 0; r < patch_height; r++) {
                        for (let c = 0; c < patch_width; c++) {
                            let val = 0;
                            for (let c_f_c = 0; c_f_c < chs; c_f_c++) {
                                for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                    for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                        if (channel_first) {
                                            val += image.get(c_f_c, r + c_f_h, c + c_f_w) * filters[f].get(c_f_h, c_f_w, c_f_c);
                                        }
                                        else {
                                            val += image.get(r + c_f_h, c + c_f_w, c_f_c) * filters[f].get(c_f_h, c_f_w, c_f_c);
                                        }
                                    }
                                }
                            }
                            if (channel_first) {
                                patch.set(f, r, c, val);
                            }
                            else {
                                patch.set(r, c, f, val);
                            }
                        }
                    }
                }
                return patch;
            }
        }
        updateWeights(l_rate) {
            for (let i = 0; i < this.filters.length; i++) {
                this.filters[i] = this.filters[i].sub(this.errorFilters[i].rotate180().mul(l_rate));
            }
            this.bias = this.bias.sub(this.errorBias.mul(l_rate));
        }
        toSavedModel() {
            return {
                filters: this.filters.map((t) => t.tensor),
                nr_filters: this.nr_filters,
                filterSize: this.filterSize,
                bias: this.bias.vector,
                shape: this.shape,
                activation: this.activationFunction.name,
                prevLayerShape: this.prevLayerShape
            };
        }
        fromSavedModel(data) {
            this.filters = data.filters.map((t) => tensor_3.default.fromJsonObject(t));
            this.nr_filters = data.nr_filters;
            this.filterSize = data.filterSize;
            this.bias = vector_6.default.fromJsonObj(data.bias);
            this.shape = data.shape;
            this.activationFunction = activations_1.default.fromName(data.activation);
            this.prevLayerShape = data.prevLayerShape;
        }
    }
    exports.default = ConvolutionLayer;
});
define("layers/dense_layer", ["require", "exports", "layers/layer", "matrix", "activations/activations", "vector", "activations/sigmoid"], function (require, exports, layer_2, matrix_8, activations_2, vector_7, sigmoid_2) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    layer_2 = __importDefault(layer_2);
    matrix_8 = __importDefault(matrix_8);
    activations_2 = __importDefault(activations_2);
    vector_7 = __importDefault(vector_7);
    sigmoid_2 = __importDefault(sigmoid_2);
    class DenseLayer extends layer_2.default {
        constructor(layerSize = 1, activation = new sigmoid_2.default()) {
            super();
            this.activationFunction = activation;
            this.layerSize = layerSize;
            this.hasGPUSupport = true;
            this.type = "dense";
        }
        buildLayer(prevLayerShape) {
            this.shape = [this.layerSize];
            this.prevLayerShape = prevLayerShape;
            this.weights = new matrix_8.default();
            this.weights.createEmptyArray(prevLayerShape[0], this.layerSize);
            this.bias = new vector_7.default(this.layerSize);
            this.weights.populateRandom();
            this.bias.populateRandom();
            this.errorWeights = new matrix_8.default();
            this.errorBias = new matrix_8.default();
            this.output_error = new matrix_8.default();
            this.activation = new matrix_8.default();
        }
        buildFFKernels(batch_size) {
            const output_shape = [this.weights.dim().c, batch_size];
            this.ff_kernel = this.gpuInstance.createKernel(function (a, w, b) {
                let sum = 0;
                for (let i = 0; i < this.constants.arr_length; i++) {
                    sum += a[this.thread.y][i] * w[i][this.thread.x];
                }
                return sum + b[this.thread.x];
            })
                .setPipeline(true)
                .setPrecision("single")
                .setConstants({ arr_length: this.weights.dim().r })
                .setDynamicOutput(false)
                .setOutput(output_shape);
            this.ff_kernel.immutable = true;
            this.act_kernel = this.gpuInstance.createKernel(this.activationFunction.normal_gpu())
                .setPipeline(true)
                .setConstants({ softmax: this.weights.dim().c })
                .setPrecision("single")
                .setDynamicOutput(false)
                .setOutput(output_shape);
            this.act_kernel.immutable = true;
        }
        buildBPKernels(length) {
            const output_shape = [this.activation.dim().c, this.activation.dim().r];
            this.bp_error_kernel = this.gpuInstance.createKernel(function (a, pW, pO) {
                let sum = 0;
                for (let i = 0; i < this.constants.mmlength; i++) {
                    sum += pO[this.thread.y][i] * pW[this.thread.x][i];
                }
                // @ts-ignore
                return sum * actv_der(a[this.thread.y][this.thread.x]);
            })
                .addFunction(this.activationFunction.derivative_gpu(), { output: output_shape })
                .setPipeline(true)
                .setPrecision("single")
                .setDynamicOutput(false)
                .setOutput(output_shape)
                .setConstants({ mmlength: length });
            this.bp_error_kernel.immutable = true;
            this.bp_error_weight_kernel = this.gpuInstance.createKernel(function (a, e) {
                let sum = 0;
                for (let i = 0; i < this.constants.arr_length; i++) {
                    sum += a[i][this.thread.y] * e[i][this.thread.x];
                }
                return sum;
            })
                .setPrecision("single")
                .setDynamicOutput(true);
            this.bp_error_weight_kernel.immutable = true;
        }
        feedForward(input, isInTraining) {
            if (this.useGpu) {
                const result = this.act_kernel(this.ff_kernel(input, this.weights.toNumberArray(), this.bias.toNumberArray()));
                this.activation = new matrix_8.default(result.toArray());
                return result;
            }
            else {
                let act;
                if (input instanceof matrix_8.default) {
                    act = input;
                }
                else {
                    act = input.activation;
                }
                const z = act.mm(this.weights);
                //console.log(z.toString(10, 6))
                z.iterate((i, j) => {
                    z.set(i, j, z.get(i, j) + this.bias.get(j));
                });
                this.activation = this.activationFunction.normal(z);
                //console.log(this.activation.toString())
            }
        }
        calculate_errors(error, input) {
        }
        backPropagation(prev_layer, next_layer) {
            if (this.useGpu) {
                let input;
                if (next_layer instanceof layer_2.default) {
                    input = next_layer.activation;
                }
                else {
                    input = next_layer;
                }
                const error = this.bp_error_kernel(this.activation.toNumberArray(), prev_layer.weights.toNumberArray(), prev_layer.output_error);
                this.output_error = error;
                this.bp_error_weight_kernel.setOutput([this.activation.dim().c, input.dim().c])
                    .setConstants({ arr_length: input.dim().r });
                const error_weights = this.bp_error_weight_kernel(input.toNumberArray(), error);
                this.errorWeights = new matrix_8.default(error_weights);
                const errorMatrix = new matrix_8.default(error.toArray());
                this.errorBias = errorMatrix.sum(0);
            }
            else {
                let dzh_dwh;
                if (next_layer instanceof layer_2.default) {
                    dzh_dwh = next_layer.activation;
                }
                else {
                    dzh_dwh = next_layer;
                }
                const deltaActv = this.activationFunction.derivative(this.activation);
                // @ts-ignore
                const error = (prev_layer.output_error.mm(prev_layer.weights.transpose())).mul(deltaActv);
                this.errorWeights = dzh_dwh.transpose().mm(error);
                this.errorBias = error.sum(0);
                this.output_error = error;
            }
        }
        updateWeights(l_rate) {
            this.weights = this.weights.sub(this.errorWeights.mul(l_rate));
            this.bias.iterate((val, i) => {
                this.bias.set(i, val - (this.errorBias.get(0, i) * l_rate));
            });
        }
        toSavedModel() {
            return {
                weights: this.weights.matrix,
                bias: this.bias.vector,
                shape: this.shape,
                activation: this.activationFunction.name
            };
        }
        fromSavedModel(data) {
            this.weights = matrix_8.default.fromJsonObject(data.weights);
            this.bias = vector_7.default.fromJsonObj(data.bias);
            this.shape = data.shape;
            this.activationFunction = activations_2.default.fromName(data.activation);
        }
    }
    exports.default = DenseLayer;
});
define("layers/dropout_layer", ["require", "exports", "layers/layer"], function (require, exports, layer_3) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    layer_3 = __importDefault(layer_3);
    class DropoutLayer extends layer_3.default {
        constructor(rate = 0.2) {
            super();
            this.rate = 0;
            this.type = "dropout";
            this.rate = rate;
        }
        buildLayer(prevLayerShape) {
            this.shape = prevLayerShape;
        }
        feedForward(input, isInTraining) {
            this.activation = input.activation;
            if (isInTraining) {
                this.activation.iterate((i, j) => {
                    if (Math.random() < this.rate) {
                        this.activation.set(i, j, 0);
                    }
                });
            }
        }
        backPropagation(prev_layer, next_layer) {
            this.weights = prev_layer.weights;
            this.output_error = prev_layer.output_error;
        }
        updateWeights(l_rate) { }
        toSavedModel() {
            return {
                rate: this.rate,
                shape: this.shape
            };
        }
        fromSavedModel(data) {
            this.shape = data.shape;
            this.rate = data.rate;
        }
    }
    exports.default = DropoutLayer;
});
define("layers/flatten_layer", ["require", "exports", "layers/layer", "tensor", "matrix"], function (require, exports, layer_4, tensor_4, matrix_9) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    layer_4 = __importDefault(layer_4);
    tensor_4 = __importDefault(tensor_4);
    matrix_9 = __importDefault(matrix_9);
    class FlattenLayer extends layer_4.default {
        constructor() {
            super(...arguments);
            this.type = "flatten";
            this.prevShape = [];
        }
        buildLayer(prevLayerShape) {
            this.prevShape = prevLayerShape;
            this.shape = [prevLayerShape.reduce((acc, val) => acc * val)];
        }
        feedForward(input, isInTraining) {
            const matrix = new matrix_9.default(input.activation.map((tensor) => tensor.vectorize(true)));
            this.activation = matrix.transpose();
        }
        backPropagation(prev_layer, next_layer) {
            let error;
            if (prev_layer.output_error instanceof matrix_9.default) {
                error = prev_layer.output_error;
            }
            else {
                error = new matrix_9.default(prev_layer.output_error.toArray());
            }
            const dout = error.mm(prev_layer.weights.transpose());
            let t = new Array(error.dim().r);
            for (let i = 0; i < t.length; i++) {
                t[i] = new tensor_4.default();
                t[i].createEmptyArray(this.prevShape[0], this.prevShape[1], this.prevShape[2]);
            }
            let [h, w, d] = this.prevShape;
            dout.iterate((n, i) => {
                const r = Math.floor(i / (w * d));
                const c = Math.floor(i / (d) - (r * w));
                const g = Math.floor(i - (c * d) - (r * w * d));
                t[n].set(r, c, g, dout.get(n, i));
            });
            this.output_error = t;
        }
        toSavedModel() {
            return {
                shape: this.prevShape
            };
        }
        fromSavedModel(data) {
            this.buildLayer(data.shape);
        }
    }
    exports.default = FlattenLayer;
});
define("losses/gradients", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class Gradients {
        static get_gradient(actvFunc, lossFunc) {
            let gradientFunc;
            if (actvFunc.name == "softmax" && lossFunc.name == "cross_entropy") {
                gradientFunc = function (input, labels) {
                    return input.sub(labels);
                };
            }
            else if (actvFunc.name == "sigmoid" && lossFunc.name == "mean_squared_error") {
                gradientFunc = function (input, labels) {
                    return input.sub(labels);
                };
            }
            return gradientFunc;
        }
    }
    exports.default = Gradients;
});
define("layers/output_layer", ["require", "exports", "matrix", "layers/dense_layer", "activations/activations", "losses/losses", "vector", "activations/sigmoid", "losses/gradients"], function (require, exports, matrix_10, dense_layer_1, activations_3, losses_1, vector_8, sigmoid_3, gradients_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    matrix_10 = __importDefault(matrix_10);
    dense_layer_1 = __importDefault(dense_layer_1);
    activations_3 = __importDefault(activations_3);
    losses_1 = __importDefault(losses_1);
    vector_8 = __importDefault(vector_8);
    sigmoid_3 = __importDefault(sigmoid_3);
    gradients_1 = __importDefault(gradients_1);
    class OutputLayer extends dense_layer_1.default {
        constructor(layerSize = 1, activation = new sigmoid_3.default()) {
            super(layerSize, activation);
            this.loss = 0;
            this.accuracy = 0;
            this.layerSize = 0;
            this.layerSize = layerSize;
            this.type = "output";
        }
        buildLayer(prevLayerShape) {
            super.buildLayer(prevLayerShape);
            this.gradientFunction = gradients_1.default.get_gradient(this.activationFunction, this.lossFunction);
        }
        backPropagationOutputLayer(labels, next_layer) {
            this.loss = labels.mul(-1).mul(this.activation.log()).sum();
            const gradient = this.gradientFunction(this.activation, labels);
            let total_acc = 0;
            let total_loss = 0;
            for (let i = 0; i < labels.dim().r; i++) {
                total_acc += this.activation.argmax(i) == labels.argmax(i) ? 1 : 0;
                total_loss += Math.abs(gradient.get(i, 0));
            }
            this.accuracy = total_acc / labels.dim().r;
            //this.loss = total_loss
            this.errorBias = gradient;
            this.output_error = gradient;
            this.errorWeights = next_layer.activation.transpose().mm(gradient);
        }
        toSavedModel() {
            return {
                weights: this.weights.matrix,
                bias: this.bias.vector,
                loss: this.lossFunction.name,
                shape: this.shape,
                activation: this.activationFunction.name
            };
        }
        fromSavedModel(data) {
            this.weights = matrix_10.default.fromJsonObject(data.weights);
            this.bias = vector_8.default.fromJsonObj(data.bias);
            this.shape = data.shape;
            this.activationFunction = activations_3.default.fromName(data.activation);
            this.lossFunction = losses_1.default.fromName(data.loss);
        }
    }
    exports.default = OutputLayer;
});
define("layers/pooling_layer", ["require", "exports", "layers/layer", "tensor"], function (require, exports, layer_5, tensor_5) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    layer_5 = __importDefault(layer_5);
    tensor_5 = __importDefault(tensor_5);
    class PoolingLayer extends layer_5.default {
        constructor(filterSize = [2, 2], stride = null, ch_first = false) {
            super();
            this.type = "pooling";
            this.prevShape = [];
            this.filterSize = [];
            this.padding = 0;
            this.stride = [];
            this.channel_first = true;
            this.poolingFunc = "max";
            this.channel_first = ch_first;
            this.filterSize = filterSize;
            this.stride = stride ? stride : filterSize;
        }
        buildLayer(prevLayerShape) {
            this.prevShape = prevLayerShape;
            let h, w, ch;
            const [f_h, f_w] = this.filterSize;
            if (this.channel_first) {
                ch = prevLayerShape[0];
                h = prevLayerShape[1];
                w = prevLayerShape[2];
            }
            else {
                h = prevLayerShape[0];
                w = prevLayerShape[1];
                ch = prevLayerShape[2];
            }
            this.shape = [
                ((h + 2 * this.padding) - f_h) / this.stride[0] + 1,
                ((w + 2 * this.padding) - f_w) / this.stride[1] + 1,
                ch
            ];
            console.log(h, (h + 2 * this.padding) - f_h / this.stride[0]);
            this.prevLayerShape = prevLayerShape;
        }
        feedForward(input, isInTraining) {
            let input_images;
            if (input instanceof layer_5.default) {
                input_images = input.activation;
            }
            else {
                input_images = input;
            }
            let h, w, ch;
            const [f_h, f_w] = this.filterSize;
            if (this.channel_first) {
                ch = this.prevLayerShape[0];
                h = this.prevLayerShape[1];
                w = this.prevLayerShape[2];
            }
            else {
                h = this.prevLayerShape[0];
                w = this.prevLayerShape[1];
                ch = this.prevLayerShape[2];
            }
            const patch_width = this.shape[1];
            const patch_height = this.shape[0];
            let new_images = [];
            for (let t = 0; t < input_images.length; t++) {
                let patch = new tensor_5.default();
                if (this.channel_first) {
                    patch.createEmptyArray(ch, patch_height, patch_width);
                }
                else {
                    patch.createEmptyArray(patch_height, patch_width, ch);
                }
                for (let f = 0; f < ch; f++) {
                    for (let r = 0; r < h; r += this.stride[0]) {
                        for (let c = 0; c < w; c += this.stride[1]) {
                            let val = [];
                            for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                    if (this.channel_first) {
                                        val.push(input_images[t].get(f, r + c_f_h, c + c_f_w));
                                    }
                                    else {
                                        val.push(input_images[t].get(r + c_f_h, c + c_f_w, f));
                                    }
                                }
                            }
                            if (this.channel_first) {
                                patch.set(f, r / this.stride[0], c / this.stride[1], Math.max(...val));
                            }
                            else {
                                patch.set(r / this.stride[0], c / this.stride[1], f, Math.max(...val));
                            }
                        }
                    }
                }
                new_images.push(patch);
            }
            this.activation = new_images;
        }
        backPropagation(prev_layer, next_layer) {
            const gradients = prev_layer.output_error;
            let input;
            if (next_layer instanceof layer_5.default) {
                input = next_layer.activation;
            }
            else {
                input = next_layer;
            }
            let t = new Array(gradients.length);
            for (let i = 0; i < t.length; i++) {
                t[i] = new tensor_5.default();
                t[i].createEmptyArray(this.prevShape[0], this.prevShape[1], this.prevShape[2]);
            }
            const [s_h, s_w] = this.stride;
            const [h, w, d] = this.prevShape;
            const [hh, ww] = this.shape;
            const [f_h, f_w] = this.filterSize;
            for (let n = 0; n < t.length; n++) {
                for (let ch = 0; ch < d; ch++) {
                    for (let r = 0; r < hh; r++) {
                        for (let c = 0; c < ww; c++) {
                            let i = -1;
                            let j = -1;
                            for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                                for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                                    if (input[n].get((r * s_h) + c_f_h, (c * s_w) + c_f_w, ch) == this.activation[n].get(r, c, ch)) {
                                        i = c_f_h;
                                        j = c_f_w;
                                        break;
                                    }
                                }
                            }
                            t[n].set((r * s_h) + i, (c * s_w) + j, ch, gradients[n].get(r, c, ch));
                        }
                    }
                }
            }
            this.output_error = t;
        }
        toSavedModel() {
            return {
                filterSize: this.filterSize,
                shape: this.shape,
                prevLayerShape: this.prevLayerShape,
                poolingFunc: this.poolingFunc,
                padding: this.padding,
                stride: this.stride
            };
        }
        fromSavedModel(data) {
            this.filterSize = data.filterSize;
            this.shape = data.shape;
            this.prevLayerShape = data.prevLayerShape;
            this.poolingFunc = data.poolingFunc;
            this.stride = data.stride;
            this.padding = data.padding;
        }
    }
    exports.default = PoolingLayer;
});
define("layers/layer_helper", ["require", "exports", "layers/conv_layer", "layers/dense_layer", "layers/dropout_layer", "layers/flatten_layer", "layers/output_layer", "activations/sigmoid", "layers/pooling_layer"], function (require, exports, conv_layer_1, dense_layer_2, dropout_layer_1, flatten_layer_1, output_layer_1, sigmoid_4, pooling_layer_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    conv_layer_1 = __importDefault(conv_layer_1);
    dense_layer_2 = __importDefault(dense_layer_2);
    dropout_layer_1 = __importDefault(dropout_layer_1);
    flatten_layer_1 = __importDefault(flatten_layer_1);
    output_layer_1 = __importDefault(output_layer_1);
    sigmoid_4 = __importDefault(sigmoid_4);
    pooling_layer_1 = __importDefault(pooling_layer_1);
    class LayerHelper {
        static fromType(type) {
            switch (type) {
                case "conv": return new conv_layer_1.default(0, [], false, new sigmoid_4.default());
                case "dense": return new dense_layer_2.default();
                case "dropout": return new dropout_layer_1.default();
                case "flatten": return new flatten_layer_1.default();
                case "output": return new output_layer_1.default();
                case "pooling": return new pooling_layer_1.default();
            }
        }
    }
    exports.LayerHelper = LayerHelper;
});
define("helpers/helper", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    class Helper {
        static timeit(func, floorIt = true) {
            return new Promise((resolve) => __awaiter(this, void 0, void 0, function* () {
                const startTime = Date.now();
                func();
                const duration = (Date.now() - startTime) / 1000.0;
                if (floorIt) {
                    resolve(Math.floor(duration));
                }
                else {
                    resolve(duration);
                }
            }));
        }
    }
    exports.default = Helper;
});
define("model", ["require", "exports", "dataset", "fs", "matrix", "vector", "gpu.js", "tensor", "layers/layer_helper", "helpers/helper", "layers/output_layer", "path"], function (require, exports, dataset_1, fs, matrix_11, vector_9, gpu_js_2, tensor_6, layer_helper_1, helper_1, output_layer_2, path_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    dataset_1 = __importDefault(dataset_1);
    fs = __importStar(fs);
    matrix_11 = __importDefault(matrix_11);
    vector_9 = __importDefault(vector_9);
    tensor_6 = __importDefault(tensor_6);
    helper_1 = __importDefault(helper_1);
    output_layer_2 = __importDefault(output_layer_2);
    path_1 = __importDefault(path_1);
    class Model {
        constructor(layers) {
            this.isBuilt = false;
            this.backlog = {
                actual_duration: 0, calculated_duration: 0, epochs: {}
            };
            this.settings = {
                USE_GPU: false,
                BACKLOG: true,
                SAVE_CHECKPOINTS: false,
                MODEL_SAVE_PATH: ""
            };
            this.model_data = {
                input_shape: [0],
                learning_rate: 0,
                last_epoch: 0
            };
            this.layers = layers;
            this.gpuInstance = new gpu_js_2.GPU();
        }
        isGpuAvailable() {
            return gpu_js_2.GPU.isGPUSupported;
        }
        build(inputShape, lossFunction, verbose = true) {
            if (!(this.layers[this.layers.length - 1] instanceof output_layer_2.default)) {
                throw "Last layer must be an OutputLayer!...";
            }
            if (!this.isGpuAvailable()) {
                console.error("GPU is not supported.. falling back on CPU.");
                this.settings.USE_GPU = false;
            }
            if (this.settings.SAVE_CHECKPOINTS && !this.settings.MODEL_SAVE_PATH) {
                console.error("No model path specified.. Turning of saving checkpoints.");
                this.settings.SAVE_CHECKPOINTS = false;
            }
            if (this.settings.BACKLOG && !this.settings.MODEL_SAVE_PATH) {
                console.error("No model path specified.. Turning of saving backlog.");
                this.settings.SAVE_CHECKPOINTS = false;
            }
            if (this.settings.MODEL_SAVE_PATH) {
                if (!fs.existsSync(this.settings.MODEL_SAVE_PATH)) {
                    fs.mkdirSync(this.settings.MODEL_SAVE_PATH);
                }
            }
            this.model_data.input_shape = inputShape;
            this.layers[0].isFirstLayer = true;
            for (let i = 0; i < this.layers.length; i++) {
                this.layers[i].setGpuInstance(this.gpuInstance);
                this.layers[i].useGpu = this.settings.USE_GPU;
                if (i == this.layers.length - 1) {
                    this.layers[i].lossFunction = lossFunction;
                }
                this.layers[i].buildLayer(i == 0 ? inputShape : this.layers[i - 1].shape);
            }
            if (verbose) {
                console.log("Successfully build model!");
            }
            this.isBuilt = true;
        }
        summary() {
            if (this.isBuilt) {
                let input = { type: "input", shape: this.model_data.input_shape, activation: "NO ACTIVATION" };
                let layer_info = this.layers.map((layer) => layer.getLayerInfo());
                let total_neurons = layer_info.map((info) => info.shape).reduce((acc, val) => {
                    return acc + val.reduce((a, s) => a * s, 1);
                }, 0);
                console.table([input, ...layer_info]);
                console.log("Total: neurons: ", total_neurons);
            }
            else {
                console.log("Model hasn't been built yet!..");
            }
        }
        train_on_batch(examples, labels) {
            if (this.settings.USE_GPU) {
                let result = examples instanceof matrix_11.default ? examples.toNumberArray() :
                    examples.map((t) => t.toNumberArray());
                let batch_size = examples instanceof matrix_11.default ? examples.dim().r :
                    examples.length;
                for (let i = 0; i < this.layers.length; i++) {
                    if (this.layers[i].hasGPUSupport) {
                        this.layers[i].buildFFKernels(batch_size);
                        result = this.layers[i].feedForward(result, true);
                    }
                    else {
                        this.layers[i].feedForward(i == 0 ? examples : this.layers[i - 1], true);
                    }
                }
                //@ts-ignore
                this.layers[this.layers.length - 1].backPropagationOutputLayer(labels, this.layers[this.layers.length - 2]);
                this.layers[this.layers.length - 1].output_error = this.layers[this.layers.length - 1].output_error.toNumberArray();
                for (let i = this.layers.length - 2; i >= 0; i--) {
                    if (this.layers[i].hasGPUSupport) {
                        this.layers[i].buildBPKernels(this.layers[i + 1].weights.dim().c);
                    }
                    let input = i == 0 ? examples : this.layers[i - 1];
                    this.layers[i].backPropagation(this.layers[i + 1], input);
                }
                for (let layer of this.layers) {
                    layer.updateWeights(this.model_data.learning_rate);
                }
                return { loss: this.layers[this.layers.length - 1].loss,
                    accuracy: this.layers[this.layers.length - 1].accuracy };
            }
            else {
                this.layers[0].feedForward(examples, true);
                for (let i = 1; i < this.layers.length; i++) {
                    this.layers[i].feedForward(this.layers[i - 1], true);
                }
                this.layers[this.layers.length - 1].backPropagationOutputLayer(labels, this.layers[this.layers.length - 2]);
                for (let i = this.layers.length - 2; i > 0; i--) {
                    this.layers[i].backPropagation(this.layers[i + 1], this.layers[i - 1]);
                }
                this.layers[0].backPropagation(this.layers[1], examples);
                for (let layer of this.layers) {
                    layer.updateWeights(this.model_data.learning_rate);
                }
                return { loss: this.layers[this.layers.length - 1].loss,
                    accuracy: this.layers[this.layers.length - 1].accuracy };
            }
        }
        train(data, epochs, learning_rate, shuffle = false, verbose = true) {
            return __awaiter(this, void 0, void 0, function* () {
                if (!this.isBuilt) {
                    throw "Model hasn't been build yet!..";
                }
                this.model_data.learning_rate = learning_rate;
                if (data instanceof dataset_1.default) {
                    console.log("Starting training...");
                    const startTime = Date.now();
                    if (data.IS_GENERATOR) {
                        const batch_count = Math.floor(data.TOTAL_EXAMPLES / data.BATCH_SIZE);
                        console.log("Total " + batch_count + " batches for " + epochs + " epochs.");
                        for (let epoch = 0; epoch < epochs; epoch++) {
                            console.log("Starting Epoch:", epoch);
                            for (let batch_id = 0; batch_id < batch_count; batch_id++) {
                                const batch = yield data.GENERATOR(batch_id);
                                const examples = new matrix_11.default(batch.map((ex) => ex.data)).transpose();
                                const labels = new matrix_11.default(batch.map((ex) => ex.label)).transpose();
                                let error = this.train_on_batch(examples, labels);
                                console.log("Error for batch: " + batch_id + " =", error);
                            }
                        }
                    }
                    else {
                        const batch_count = Math.floor(data.size() / data.BATCH_SIZE);
                        for (let epoch = 1; epoch <= epochs; epoch++) {
                            console.log("Starting Epoch:", epoch, "/", epochs);
                            if (shuffle) {
                                data.shuffle();
                            }
                            const epoch_data = {
                                total_loss: 0,
                                total_accuracy: 0,
                                batches: [],
                                calculated_duration: 0,
                                actual_duration: 0
                            };
                            const epoch_startTime = Date.now();
                            for (let batch_id = 0; batch_id < batch_count; batch_id++) {
                                let batch = data.getBatch(batch_id);
                                let examples;
                                let b_loss = 0;
                                let b_acc = 0;
                                let exampleData = batch.map((ex) => ex.data);
                                const labels = new matrix_11.default(batch.map((ex) => ex.label)).transpose();
                                if (data.DATA_STRUCTURE == vector_9.default) {
                                    examples = new matrix_11.default(batch.map((ex) => ex.data)).transpose();
                                }
                                else if (data.DATA_STRUCTURE == tensor_6.default) {
                                    examples = exampleData;
                                }
                                const seconds = yield helper_1.default.timeit(() => {
                                    let { loss, accuracy } = this.train_on_batch(examples, labels);
                                    b_loss = loss;
                                    b_acc = accuracy;
                                }, false);
                                epoch_data.batches.push({ accuracy: b_acc, loss: b_loss, time: seconds });
                                epoch_data.total_loss += b_loss;
                                epoch_data.total_accuracy += b_acc;
                                epoch_data.calculated_duration += seconds;
                                this.backlog.calculated_duration += seconds;
                                this.backlog["epoch_" + epoch] = epoch_data;
                                this.saveBacklog();
                                console.log("Batch:", (batch_id + 1), "/", batch_count, "Loss =", b_loss, ", Acc = ", b_acc, "| Time:", seconds, "seconds");
                            }
                            epoch_data.actual_duration = (Date.now() - epoch_startTime) / 1000;
                            this.backlog.epochs["epoch_" + epoch] = epoch_data;
                            console.log("Loss: TOT", epoch_data.total_loss, "AVG", epoch_data.total_loss / batch_count, "| Accuracy:", epoch_data.total_accuracy / batch_count, "| Total time:", epoch_data.actual_duration, "/", epoch_data.calculated_duration);
                            this.saveBacklog();
                            this.model_data.last_epoch = epoch;
                            if (this.settings.SAVE_CHECKPOINTS) {
                                this.save("model_checkpoint_" + epoch + ".json");
                            }
                        }
                    }
                    console.log("Done..");
                    const duration = (Date.now() - startTime) / 1000;
                    this.backlog.actual_duration = duration;
                    console.log("Duration: " + duration + " seconds");
                    this.saveBacklog();
                }
                else {
                    let exampleData = data.map((ex) => ex.data);
                    let examples = exampleData[0] instanceof vector_9.default ? new matrix_11.default(exampleData) : exampleData;
                    let labels = new matrix_11.default(data.map((ex) => ex.label)).transpose();
                    for (let epoch = 0; epoch < epochs; epoch++) {
                        console.log(this.train_on_batch(examples, labels));
                    }
                }
            });
        }
        saveBacklog() {
            if (this.settings.BACKLOG) {
                const path = path_1.default.join(this.settings.MODEL_SAVE_PATH, "backlog.json");
                fs.writeFileSync(path, JSON.stringify(this.backlog));
            }
        }
        predict(data) {
            if (!this.isBuilt) {
                throw "Model hasn't been build yet!..";
            }
            let exampleMatrix;
            if (data instanceof vector_9.default) {
                exampleMatrix = new matrix_11.default([data]).transpose();
            }
            else {
                exampleMatrix = [data];
            }
            this.layers[0].feedForward(exampleMatrix, false);
            for (let i = 1; i < this.layers.length; i++) {
                this.layers[i].feedForward(this.layers[i - 1], false);
            }
            return this.layers[this.layers.length - 1].activation;
        }
        save(model_path = "model.json") {
            const modelObj = {
                model_data: this.model_data,
                settings: this.settings,
                layers: {}
            };
            for (let i = 0; i < this.layers.length; i++) {
                modelObj.layers[`layer_${i}`] = {
                    type: this.layers[i].type,
                    info: this.layers[i].toSavedModel()
                };
            }
            const path = path_1.default.join(this.settings.MODEL_SAVE_PATH, model_path);
            fs.writeFileSync(path, JSON.stringify(modelObj));
        }
        load(path, verbose = true) {
            const modelObj = JSON.parse(fs.readFileSync(path, { encoding: "UTF-8" }));
            this.model_data = modelObj.model_data;
            this.settings = modelObj.settings;
            const layer_keys = Object.keys(modelObj.layers).sort();
            this.layers = [];
            if (!this.isGpuAvailable()) {
                console.error("GPU is not supported.. falling back on CPU.");
                this.settings.USE_GPU = false;
            }
            for (let layer_key of layer_keys) {
                let layer = layer_helper_1.LayerHelper.fromType(modelObj.layers[layer_key].type);
                layer.fromSavedModel(modelObj.layers[layer_key].info);
                layer.setGpuInstance(this.gpuInstance);
                layer.useGpu = this.settings.USE_GPU;
                this.layers.push(layer);
            }
            this.layers[0].isFirstLayer = true;
            this.isBuilt = true;
            if (verbose) {
                console.log("Successfully build model!");
            }
        }
    }
    exports.default = Model;
});
define("helpers/matrix_helper", ["require", "exports", "matrix"], function (require, exports, matrix_12) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    matrix_12 = __importDefault(matrix_12);
    class MatrixHelper {
        static row_reduction(matrix) {
            let m = matrix.copy();
            let h = 0;
            let k = 0;
            let swapArray = [];
            while (h < m.dim().r && k < m.dim().c) {
                let i_max = m.argmax(h, false);
                if (m.get(i_max) == 0) {
                    k += 1;
                }
                else {
                    const tempRow = m.matrix[h];
                    m.matrix[h] = m.matrix[i_max];
                    m.matrix[i_max] = tempRow;
                    swapArray.push([h, i_max]);
                    for (let i = h + 1; i < m.dim().r; i++) {
                        let f = m.get(i, k) / m.get(h, k);
                        m.set(i, k, 0);
                        for (let j = k + 1; j < m.dim().c; j++) {
                            m.set(i, j, m.get(i, j) - (m.get(i, j) * f));
                        }
                    }
                    h++;
                    k++;
                }
            }
            for (let [i, j] of swapArray) {
                const tempRow = m.matrix[i];
                m.matrix[i] = m.matrix[j];
                m.matrix[j] = tempRow;
            }
            return m;
        }
        /*public static diagonalize(m: Matrix): Matrix {
            if(m.dim().c == m.dim().r) {
                let zerosCount = new Vector(m.dim().r)
                m.iterate((i,j) => {zerosCount.set(i, zerosCount.get(i) + (m.get(i,j) == 0? 1 : 0))})
    
                console.log(zerosCount.toString())
                for(let i = 1; i < m.dim().r; i++) {
                    const smallestIndex = zerosCount.argmax()
                    zerosCount.set(smallestIndex, -1)
                    const secondRowIndex = zerosCount.argmax();
                }
            } else {
                return new Matrix()
            }
        }*/
        static linear_least_squares(x, y) {
            let A = new matrix_12.default();
            A.createEmptyArray(x.size(), 2);
            A.matrix.forEach((val, index) => {
                A.set(index, 0, 1);
                A.set(index, 1, x.get(index));
            });
            const VL = A.transpose().mm(A);
            const HL = A.transpose().mm(y);
            let xV = VL.inv().mm(HL);
            console.log(xV.toString());
            //const reducedVL = this.row_reduction(VL)
            //let zerosCount = new Vector(reducedVL.dim().r)
            //reducedVL.iterate((i,j) => {zerosCount.set(i, zerosCount.get(i) + (reducedVL.get(i,j) == 0? 1 : 0))})
        }
        static linear_least_squares_pol(x, y) {
            let A = new matrix_12.default();
            A.createEmptyArray(x.size(), 2);
            A.matrix.forEach((val, index) => {
                A.set(index, 0, 1);
                A.set(index, 1, x.get(index));
            });
            const VL = A.transpose().mm(A);
            const HL = A.transpose().mm(y);
            let xV = VL.inv().mm(HL);
            console.log(xV.toString());
            //const reducedVL = this.row_reduction(VL)
            //let zerosCount = new Vector(reducedVL.dim().r)
            //reducedVL.iterate((i,j) => {zerosCount.set(i, zerosCount.get(i) + (reducedVL.get(i,j) == 0? 1 : 0))})
        }
    }
    exports.default = MatrixHelper;
});
define("layers/batch_norm_layer", ["require", "exports", "layers/layer", "matrix"], function (require, exports, layer_6, matrix_13) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    layer_6 = __importDefault(layer_6);
    matrix_13 = __importDefault(matrix_13);
    class BatchNormLayer extends layer_6.default {
        constructor(momentum = 0.9) {
            super();
            this.cache = {};
            this.momentum = momentum;
            this.type = "batch_norm";
        }
        buildLayer(prevLayerShape) {
            const [D] = prevLayerShape;
            console.log(prevLayerShape);
            this.shape = prevLayerShape;
            this.running_mean = new matrix_13.default();
            this.running_mean.createEmptyArray(1, D);
            this.running_var = new matrix_13.default();
            this.running_var.createEmptyArray(1, D);
            this.weights = new matrix_13.default();
            this.weights.createEmptyArray(1, D);
            this.weights.populateRandom();
            this.bias = new matrix_13.default();
            this.bias.createEmptyArray(1, D);
            this.bias.populateRandom();
        }
        feedForward(input, isInTraining) {
            let act;
            if (input instanceof matrix_13.default) {
                act = input;
            }
            else {
                act = input.activation;
            }
            const N = act.dim().r;
            const mean = act.mean(0);
            const diff = act.sub(mean.repeat(0, N));
            const variance = diff.pow(2).mean(0);
            if (isInTraining) {
                console.log(act.toString());
                const xhat = diff.div(variance.sqrt().repeat(0, N).add(Math.pow(10, -5)));
                this.activation = this.weights.repeat(0, N).mul(xhat).add(this.bias.repeat(0, N));
                this.running_mean = this.running_mean.mul(this.momentum).add(mean.mul(1 - this.momentum));
                this.running_var = this.running_var.mul(this.momentum).add(variance.mul(1 - this.momentum));
                this.cache = { variance, diff, xhat };
            }
        }
        backPropagation(prev_layer, next_layer) {
            let error;
            if (prev_layer.output_error instanceof matrix_13.default) {
                error = prev_layer.output_error;
            }
            else {
                error = new matrix_13.default(prev_layer.output_error.toArray());
            }
            let X;
            if (next_layer instanceof matrix_13.default) {
                X = next_layer;
            }
            else {
                X = next_layer.activation;
            }
            const { variance, diff, xhat } = this.cache;
            const dout = error.mm(prev_layer.weights.transpose());
            const N = dout.dim().r;
            const std_inv = variance.sqrt().inv_el(Math.pow(10, -8));
            const dX_norm = dout.mul(this.weights.repeat(0, N));
            const dVar = dX_norm.mul(diff).sum(0).mul(-0.5).mul(std_inv.pow(3));
            const dMean = dX_norm.mul(std_inv.mul(-1).repeat(0, N)).sum(0).add(dVar.mul(diff.mul(-2).mean(0)));
            this.output_error = dX_norm.mul(std_inv.repeat(0, N)).add(dVar.repeat(0, N).mul(2).mul(diff).div(N)).add(dMean.div(N).repeat(0, N));
            this.errorWeights = dout.mul(xhat).sum(0);
            this.errorBias = dout.sum(0);
        }
        updateWeights(l_rate) {
            this.weights = this.weights.sub(this.errorWeights.mul(l_rate));
            this.bias = this.bias.sub(this.errorBias.mul(l_rate));
        }
    }
    exports.default = BatchNormLayer;
});
define("linguistics/csv_parser", ["require", "exports", "fs"], function (require, exports, fs_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    fs_1 = __importDefault(fs_1);
    class CsvParser {
        static parse(data, isPath = false) {
            let content;
            if (isPath) {
                content = fs_1.default.readFileSync(data, { encoding: "utf-8" });
            }
            else {
                content = data;
            }
            const lines = content.split("\n");
            return lines.map((line) => line.trim().split("\t").map((cell) => {
                const n = parseFloat(cell.trim());
                return isFinite(n) ? n : cell.trim();
            }));
        }
        static filterColumns(data, columns) {
            return data.map((line) => {
                return line.filter((_, index) => columns.includes(index));
            });
        }
    }
    exports.default = CsvParser;
});
define("linguistics/suffixes", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.suffixes = [
        "-a",
        "-an",
        "-al",
        "-ans",
        "-aktig",
        "-ar",
        "-artad",
        "-ell",
        "-ens",
        "-gon",
        "-het",
        //"-i",
        "-ia",
        "-ibel",
        "-id",
        "-ik",
        "-il",
        "-in",
        "-ing",
        "-ion",
        "-is",
        "-isk",
        "-ism",
        "-ist",
        "-iv",
        "-ligen",
        "-logi",
        "-ment",
        "-naut",
        "-sam",
        "-skap"
    ];
});
define("linguistics/tokenizer", ["require", "exports", "linguistics/suffixes", "helpers/array_helper", "fs"], function (require, exports, suffixes_1, array_helper_2, fs_2) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    array_helper_2 = __importDefault(array_helper_2);
    fs_2 = __importDefault(fs_2);
    class Tokenizer {
        constructor() {
            this.vocab = {};
        }
        createVocabulary(sentences) {
            const sents = sentences.map((sentence) => sentence.trim().split(" "));
            const single_words = array_helper_2.default.delete_doublets(array_helper_2.default.flatten(sents));
            const vocab = array_helper_2.default.flatten(single_words.map((word) => {
                const suffix = suffixes_1.suffixes.filter((suff) => word.endsWith(suff.replace("-", "")));
                if (suffix.length > 0) {
                    return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix];
                }
                else {
                    return word;
                }
            }));
            this.vocab = vocab.sort().reduce((acc, token, index) => {
                acc[token.toString()] = index;
                return acc;
            }, {});
        }
        loadVocabulary(path) {
            this.vocab = JSON.parse(fs_2.default.readFileSync(path, { encoding: "utf-8" }));
        }
        saveVocabulary(path) {
            fs_2.default.writeFileSync(path, JSON.stringify(this.vocab));
        }
        tokenize(sentence) {
            return array_helper_2.default.flatten(sentence.split(" ").map((word) => {
                const suffix = suffixes_1.suffixes.filter((suff) => word.endsWith(suff.replace("-", "")));
                if (suffix.length > 0) {
                    return [word.substr(0, word.lastIndexOf(suffix[0].replace("-", ""))), suffix];
                }
                else {
                    return word;
                }
            })).map((token) => this.vocab[token]);
        }
    }
    exports.default = Tokenizer;
});
define("visualizer/data_handler", ["require", "exports", "fs"], function (require, exports, fs_3) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    fs_3 = __importDefault(fs_3);
    class DataHandler {
        constructor(pubSub, path) {
            this.pubSub = pubSub;
            this.watchPath = path;
            this.loadData();
        }
        loadData() {
            this.data = JSON.parse(fs_3.default.readFileSync(this.watchPath, { encoding: "utf-8" }));
        }
        startWatcher() {
            if (!fs_3.default.existsSync(this.watchPath)) {
                throw "Backlog file doesn't exists... Aborting!";
            }
            fs_3.default.watchFile(this.watchPath, {}, (stats) => {
                console.log("Backlog updated.");
                this.loadData();
            });
            console.log("Started backlog watcher!");
        }
        getBatches() {
            return Object.keys(this.data.epochs).reduce((acc, epoch) => {
                acc.push(...this.data.epochs[epoch].batches.map((batch, index) => {
                    batch["id"] = index;
                    batch["epoch_id"] = parseInt(epoch.substr(epoch.lastIndexOf("_") + 1));
                    return batch;
                }));
                return acc;
            }, []);
        }
        getBatch(epoch_id, batch_id) {
            return this.getBatches().filter((batch) => batch.id == batch_id && batch.epoch_id == epoch_id)[0];
        }
        parseEpoch(epoch) {
            const epoch_id = parseInt(epoch.substr(epoch.lastIndexOf("_") + 1));
            const data = Object.create(this.data.epochs[epoch]);
            data.batches = data.batches.map((batch, index) => {
                batch["id"] = index;
                batch["epoch_id"] = epoch_id;
                return batch;
            });
            data["accuracy"] = this.data.epochs[epoch].total_accuracy / this.data.epochs[epoch].batches.length;
            data["loss"] = this.data.epochs[epoch].total_loss / this.data.epochs[epoch].batches.length;
            data["id"] = epoch_id;
            return data;
        }
        getEpochs() {
            return Object.keys(this.data.epochs).map((epoch) => {
                return this.parseEpoch(epoch);
            });
        }
        getEpoch(epoch_id) {
            return this.parseEpoch("epoch_" + epoch_id);
        }
    }
    exports.default = DataHandler;
});
define("visualizer/visualizer", ["require", "exports", "apollo-server", "graphql-tools", "visualizer/data_handler"], function (require, exports, apollo_server_1, graphql_tools_1, data_handler_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    data_handler_1 = __importDefault(data_handler_1);
    class Visualizer {
        constructor(path) {
            this.PORT = 3000;
            this.pubsub = new apollo_server_1.PubSub();
            this.data_handler = new data_handler_1.default(this.pubsub, path);
            const typeDefs = apollo_server_1.gql(`
              type Epoch {
                id: Float
                accuracy: Float
                total_accuracy: Float
                loss: Float
                total_loss: Float
                actual_duration: Float
                calculated_duration: Float
                batches: [Batch]
              }
              
              type Batch {
                id: Float
                epoch_id: Float
                accuracy: Float
                loss: Float
                time: Float
              }
              
              type Query {
                epochs:[Epoch]
                epoch(id: Float): Epoch
                batches: [Batch]
                batch(id: Float, epoch_id: Float): Batch
              }
              
              type Subscription {
                new_batch: Batch
              }
            `);
            const schema = graphql_tools_1.makeExecutableSchema({
                typeDefs, resolvers: {
                    Query: {
                        batches: () => {
                            return this.data_handler.getBatches();
                        },
                        batch: (parent, args, context, info) => {
                            return this.data_handler.getBatch(args.epoch_id, args.id);
                        },
                        epochs: () => {
                            return this.data_handler.getEpochs();
                        },
                        epoch: (parent, args, context, info) => {
                            return this.data_handler.getEpoch(args.id);
                        }
                    },
                    Subscription: {
                        new_batch: {
                            subscribe: () => this.pubsub.asyncIterator("new_batch"),
                        },
                    }
                }
            });
            this.server = new apollo_server_1.ApolloServer({ schema });
        }
        run() {
            console.log("Starting server...");
            this.data_handler.startWatcher();
            this.server.listen({ port: this.PORT }).then(({ url }) => {
                console.log(`Visualizer server ready at ${url}`);
            });
        }
    }
    exports.default = Visualizer;
});

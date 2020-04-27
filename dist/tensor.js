"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vector_1 = __importDefault(require("./vector"));
const matrix_1 = __importDefault(require("./matrix"));
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
        const v = new vector_1.default(this.count());
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
                cols.push(new vector_1.default(v));
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

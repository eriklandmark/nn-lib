"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class Tensor {
    constructor(v = [], shape = false) {
        this.t = [];
        this.shape = [];
        this.dim = 0;
        if (shape) {
            this.shape = v;
            this.createFromShape(this.shape);
            this.dim = this.shape.length;
        }
        else {
            if (v.length == 0) {
                this.shape = [0];
            }
            else {
                this.calculateShape(v);
                this.createTensor(v);
            }
        }
        /*
        return new Proxy(this, {
            get(target, prop) {
                if (Number(prop) == prop && !(prop in target)) {
                    if (typeof target.t[prop] == "number" && !isFinite(target.t[prop])) {
                        console.trace()
                        throw "Getting an NaN..."
                    }
                    return target.t[prop];
                }
                return target[prop];
            }, set(target, prop: PropertyKey, value: any): boolean {
                if (Number(prop) == prop && !(prop in target)) {
                    if (typeof target.t[prop] != "number") {
                        console.trace()
                        throw "Cannot set array to number..."
                    }
                    if (!isFinite(value)) {
                        console.trace()
                        throw "Setting an NaN..."
                    }

                    return target.t[prop] = value
                }
            }
        });*/
    }
    createTensor(v) {
        if (v instanceof Float64Array) {
            this.t = v;
        }
        else if (typeof v[0] == "number") {
            this.t = Float64Array.from(v);
        }
        else {
            if (this.shape.length == 2) {
                this.t = [];
                for (let i = 0; i < this.shape[0]; i++) {
                    this.t.push(Float64Array.from(v[i]));
                }
            }
            else if (this.shape.length == 3) {
                this.t = [];
                for (let i = 0; i < this.shape[0]; i++) {
                    this.t.push([]);
                    for (let j = 0; j < this.shape[1]; j++) {
                        this.t[i].push(Float64Array.from(v[i][j]));
                    }
                }
            }
            else if (this.shape.length == 4) {
                this.t = [];
                for (let i = 0; i < this.shape[0]; i++) {
                    this.t.push([]);
                    for (let j = 0; j < this.shape[1]; j++) {
                        this.t[i].push([]);
                        for (let k = 0; k < this.shape[2]; k++) {
                            this.t[i][j].push(Float64Array.from(v[i][j][k]));
                        }
                    }
                }
            }
        }
    }
    createFromShape(shape) {
        this.shape = shape;
        if (shape.length == 1) {
            this.t = new Array(shape[0]).fill(0);
        }
        else if (shape.length == 2) {
            this.t = [];
            for (let i = 0; i < shape[0]; i++) {
                this.t.push(new Float64Array(shape[1]).fill(0));
            }
        }
        else if (shape.length == 3) {
            this.t = [];
            for (let i = 0; i < shape[0]; i++) {
                this.t.push([]);
                for (let j = 0; j < shape[1]; j++) {
                    this.t[i].push(new Float64Array(shape[2]).fill(0));
                }
            }
        }
        else if (shape.length == 4) {
            this.t = [];
            for (let i = 0; i < shape[0]; i++) {
                this.t.push([]);
                for (let j = 0; j < shape[1]; j++) {
                    this.t[i].push([]);
                    for (let k = 0; k < shape[2]; k++) {
                        this.t[i][j].push(new Float64Array(shape[3]).fill(0));
                    }
                }
            }
        }
    }
    calculateShape(arr) {
        let shape = [];
        const f = (v) => {
            shape.push(v.length);
            if (!(v instanceof Float64Array || typeof v[0] == "number")) {
                f(v[0]);
            }
        };
        if (arr instanceof Float64Array || typeof arr[0] == "number") {
            this.shape = [arr.length];
        }
        else {
            shape.push(arr.length);
            f(arr[0]);
            this.shape = shape;
        }
        this.dim = this.shape.length;
    }
    get(pos) {
        if (pos.length == 1) {
            if (!isFinite(this.t[pos[0]])) {
                console.trace();
                throw "Getting an NaN... (" + this.t[pos[0]] + ")";
            }
            return this.t[pos[0]];
        }
        else if (pos.length == 2) {
            if (!isFinite(this.t[pos[0]][pos[1]])) {
                console.trace();
                throw "Getting an NaN... (" + this.t[pos[0]][pos[1]] + ")";
            }
            return this.t[pos[0]][pos[1]];
        }
        else if (pos.length == 3) {
            if (!isFinite(this.t[pos[0]][pos[1]][pos[2]])) {
                console.trace();
                throw "Getting an NaN... (" + this.t[pos[0]][pos[1]][pos[2]] + ")";
            }
            return this.t[pos[0]][pos[1]][pos[2]];
        }
        else if (pos.length == 4) {
            if (!isFinite(this.t[pos[0]][pos[1]][pos[2]][pos[3]])) {
                console.trace();
                throw "Getting an NaN... (" + this.t[pos[0]][pos[1]][pos[2]][pos[3]] + ")";
            }
            return this.t[pos[0]][pos[1]][pos[2]][pos[3]];
        }
    }
    set(pos, v) {
        if (!isFinite(v) || isNaN(v)) {
            console.trace();
            throw "Number is NaN... (" + v + ")";
        }
        if (this.dim == 1) {
            this.t[pos[0]] = v;
        }
        else if (this.dim == 2) {
            this.t[pos[0]][pos[1]] = v;
        }
        else if (this.dim == 3) {
            this.t[pos[0]][pos[1]][pos[2]] = v;
        }
        else if (this.dim == 4) {
            this.t[pos[0]][pos[1]][pos[2]][pos[3]] = v;
        }
    }
    count() {
        return this.shape.reduce((acc, d) => acc * d, 1);
    }
    ;
    static toCategorical(index, size) {
        const v = new Tensor([size], true);
        v.t[index] = 1;
        return v;
    }
    static fromJsonObject(obj) {
        if (obj.length == 0) {
            return new Tensor();
        }
        else if (typeof obj["0"] == "number") {
            return new Tensor(Object.keys(obj).map((item, index) => {
                return obj[index.toString()];
            }));
        }
        else if (typeof obj["0"]["0"] == "number") {
            return new Tensor(obj.map((row) => {
                return Object.keys(row).map((item) => row[item]);
            }));
        }
        else if (typeof obj["0"]["0"]["0"] == "number") {
            return new Tensor(obj.map((row) => {
                return row.map((col) => {
                    return Object.keys(col).map((item) => col[item]);
                });
            }));
        }
        else if (typeof obj["0"]["0"]["0"]["0"] == "number") {
            return new Tensor(obj.map((row) => {
                return row.map((col) => {
                    return col.map((depth) => {
                        return Object.keys(depth).map((item) => depth[item]);
                    });
                });
            }));
        }
    }
    equalShape(t) {
        if (this.dim !== t.dim)
            return false;
        for (let i = 0; i < this.dim; i++) {
            if (this.shape[i] !== t.shape[i]) {
                return false;
            }
        }
        return true;
    }
    toNumberArray() {
        if (this.dim == 1) {
            return [].slice.call(this.t);
        }
        else if (this.dim == 2) {
            return this.t.map((floatArray) => [].slice.call(floatArray));
        }
        else if (this.dim == 3) {
            return this.t.map((array) => array.map((floatArray) => [].slice.call(floatArray)));
        }
    }
    iterate(func, use_pos = false, channel_first = false) {
        if (this.dim == 1) {
            this.t.forEach((_, index) => {
                if (use_pos) {
                    func([index]);
                }
                else {
                    func(index);
                }
            });
        }
        else if (this.dim == 2) {
            for (let i = 0; i < this.shape[0]; i++) {
                for (let j = 0; j < this.shape[1]; j++) {
                    if (use_pos) {
                        func([i, j]);
                    }
                    else {
                        func(i, j);
                    }
                }
            }
        }
        else if (this.dim == 3) {
            if (channel_first) {
                for (let k = 0; k < this.shape[2]; k++) {
                    for (let i = 0; i < this.shape[0]; i++) {
                        for (let j = 0; j < this.shape[1]; j++) {
                            if (use_pos) {
                                func([i, j, k]);
                            }
                            else {
                                func(i, j, k);
                            }
                        }
                    }
                }
            }
            else {
                for (let i = 0; i < this.shape[0]; i++) {
                    for (let j = 0; j < this.shape[1]; j++) {
                        for (let k = 0; k < this.shape[2]; k++) {
                            if (use_pos) {
                                func([i, j, k]);
                            }
                            else {
                                func(i, j, k);
                            }
                        }
                    }
                }
            }
        }
        else if (this.dim == 4) {
            for (let i = 0; i < this.shape[0]; i++) {
                for (let j = 0; j < this.shape[1]; j++) {
                    for (let k = 0; k < this.shape[2]; k++) {
                        for (let l = 0; l < this.shape[3]; l++) {
                            if (use_pos) {
                                func([i, j, k, l]);
                            }
                            else {
                                func(i, j, k, l);
                            }
                        }
                    }
                }
            }
        }
    }
    numberToString(nr, precision = 5, autoFill = false) {
        const expStr = nr.toExponential();
        return (+expStr.substr(0, expStr.lastIndexOf("e"))).toPrecision(precision)
            + expStr.substr(expStr.lastIndexOf("e")) +
            (autoFill ? " ".repeat(4 - expStr.substr(expStr.lastIndexOf("e")).length) : "");
    }
    toString(max_rows = 10, precision = 3) {
        if (this.dim == 0) {
            return "Tensor: []";
        }
        else if (this.dim == 1) {
            return this.t.reduce((acc, v) => {
                acc += `    ${this.numberToString(v, precision, true)}\n`;
                return acc;
            }, `Tensor: ${this.shape[0]} [\n`) + " ]";
        }
        else if (this.dim == 2) {
            return this.t.slice(0, Math.min(max_rows, this.t.length)).reduce((acc, i) => {
                acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s, i) => {
                    s += " "; //.repeat(Math.max(maxCharCount - i.toPrecision(precision).length, 1))
                    s += this.numberToString(i, precision, true);
                    return s;
                }, "    ");
                acc += i.length > max_rows ? "  ... +" + (i.length - max_rows) + " elements\n" : "\n";
                return acc;
            }, `Tensor: ${this.shape[0]}x${this.shape[1]} [\n`) + (this.t.length > max_rows ?
                "    ... +" + (this.t.length - max_rows) + " rows \n]" : " ]");
        }
        else if (this.dim == 3) {
            let maxCharCount = 0;
            this.iterate((i, j, k) => {
                let val = this.t[i][j][k].toString();
                if (val.length > maxCharCount)
                    maxCharCount = val.length;
            });
            maxCharCount = Math.min(maxCharCount, 7);
            let string = `Tensor: ${this.shape[0]}x${this.shape[1]}x${this.shape[2]} [\n`;
            for (let d = 0; d < this.shape[2]; d++) {
                string += this.t.slice(0, Math.min(max_rows, this.t.length)).reduce((acc, i) => {
                    acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s, j) => {
                        const v = this.numberToString(j[d], precision, true);
                        s += " "; //.repeat(Math.max(maxCharCount - j[d].toString().length, 0))
                        s += v; //j[d].toString().substr(0, maxCharCount) + " ";
                        return s;
                    }, "    ");
                    acc += i.length > max_rows ? " ... +" + (i.length - max_rows) + " elements\n" : "\n";
                    return acc;
                }, "") + (this.t.length > max_rows ?
                    "    ... +" + (this.t.length - max_rows) + " rows \n" : "\n");
            }
            return string + "]";
        }
    }
    print(max_rows = 10, precision = 3) {
        console.log(this.toString(max_rows, precision));
    }
    copy(full = true) {
        if (this.shape == [0]) {
            return new Tensor();
        }
        else {
            let t = new Tensor(this.shape, true);
            if (full) {
                t.iterate((pos) => t.set(pos, this.get(pos)), true);
            }
            return t;
        }
    }
    populateRandom(seed = null) {
        /*if (seed) {
            var x = Math.sin(seed++) * 10000;
            return x - Math.floor(x);
        }*/
        this.iterate((pos) => {
            this.set(pos, Math.random() * 2 - 1);
        }, true);
    }
    empty() {
        return this.shape[0] == 0 || this.shape[1] == 0 || this.shape[2] == 0 || this.shape[3] == 0;
    }
    vectorize(channel_first = false) {
        const t = new Tensor([this.count()], true);
        let index = 0;
        if (this.dim == 1) {
            return this.copy(true);
        }
        else {
            this.iterate((pos) => {
                t.t[index] = this.get(pos);
                index += 1;
            }, true);
        }
        return t;
    }
    div(v, safe = false) {
        let t = this.copy(false);
        if (v instanceof Tensor) {
            if (!this.equalShape(v)) {
                console.trace();
                throw "Tensor Division: Not the same shape";
            }
            this.iterate((pos) => {
                const n = safe ? v.get(pos) + Math.pow(10, -7) : v.get(pos);
                t.set(pos, this.get(pos) / n);
            }, true);
        }
        else {
            const n = safe ? v + Math.pow(10, -7) : v;
            this.iterate((pos) => {
                t.set(pos, this.get(pos) / n);
            }, true);
        }
        return t;
    }
    mul(v) {
        let t = this.copy(false);
        if (v instanceof Tensor) {
            if (!this.equalShape(v)) {
                console.trace();
                throw "Tensor Multiplication: Not the same shape";
            }
            this.iterate((pos) => {
                t.set(pos, this.get(pos) * v.get(pos));
            }, true);
        }
        else {
            this.iterate((pos) => {
                t.set(pos, this.get(pos) * v);
            }, true);
        }
        return t;
    }
    sub(v) {
        let t = this.copy(false);
        if (v instanceof Tensor) {
            if (!this.equalShape(v)) {
                console.trace();
                throw "Tensor Subtraction: Not the same shape";
            }
            this.iterate((pos) => {
                t.set(pos, this.get(pos) - v.get(pos));
            }, true);
        }
        else {
            this.iterate((pos) => {
                t.set(pos, this.get(pos) - v);
            }, true);
        }
        return t;
    }
    add(v) {
        let t = this.copy(false);
        if (v instanceof Tensor) {
            if (!this.equalShape(v)) {
                console.trace();
                throw "Tensor Addition: Not the same shape";
            }
            this.iterate((pos) => {
                t.set(pos, this.get(pos) + v.get(pos));
            }, true);
        }
        else {
            this.iterate((pos) => {
                t.set(pos, this.get(pos) + v);
            }, true);
        }
        return t;
    }
    pow(v) {
        let t = this.copy(false);
        this.iterate((pos) => {
            t.set(pos, Math.pow(this.get(pos), v));
        }, true);
        return t;
    }
    sqrt() {
        let t = this.copy(false);
        this.iterate((pos) => {
            t.set(pos, Math.sqrt(this.get(pos)));
        }, true);
        return t;
    }
    exp(base = null) {
        let t = this.copy(false);
        this.iterate((pos) => {
            t.set(pos, base ? Math.pow(base, this.get(pos)) : Math.exp(this.get(pos)));
        }, true);
        return t;
    }
    inv_el(eps = Math.pow(10, -7)) {
        let t = this.copy(false);
        this.iterate((pos) => {
            t.set(pos, this.get(pos) == 0 ? 1 / (this.get(pos) + eps) : 1 / this.get(pos));
        }, true);
        return t;
    }
    inv() {
        if (this.shape[0] == 1 && this.shape[1] == 1) {
            return new Tensor([[1 / this.t[0][0]]]);
        }
        else if (this.shape[0] == 2 && this.shape[1] == 2) {
            return new Tensor([
                [this.t[1][1], -this.t[0][1]],
                [-this.t[1][0], this.t[0][0]]
            ]).mul(1 / ((this.t[0][0] * this.t[1][1]) - (this.t[0][1] * this.t[1][0])));
        }
    }
    fill(scalar) {
        let t = this.copy(false);
        this.iterate((pos) => {
            t.set(pos, scalar);
        }, true);
        return t;
    }
    log() {
        let t = this.copy(false);
        this.iterate((pos) => {
            t.set(pos, Math.log(this.get(pos)));
        }, true);
        return t;
    }
    dot(b) {
        if (this.dim == 2) {
            if (b.dim == 1) {
                if (b.shape[0] != this.shape[1]) {
                    console.trace();
                    throw "Matrix Multiplication (Vector): Wrong dimension..\n" +
                        "This: [ " + this.shape[0] + " , " + this.shape[1] + " ] | Other: [ " + b.shape[0] + " ]";
                }
                const t = new Tensor([this.shape[1]], true);
                for (let i = 0; i < this.shape[1]; i++) {
                    t[i] = this.t[i].reduce((acc, val, k) => acc + (val * t[k]), 0);
                }
                return t;
            }
            else if (b.dim == 2) {
                if (this.shape[1] != b.shape[0]) {
                    console.trace();
                    throw "Matrix Multiplication (Matrix): Wrong dimension..\n" +
                        "This: [ " + this.shape[0] + " , " + this.shape[1] + " ] | Other: [ " + b.shape[0] + " , " + b.shape[1] + " ]";
                }
                const t = new Tensor([this.shape[0], b.shape[1]], true);
                for (let i = 0; i < this.shape[0]; i++) {
                    for (let j = 0; j < this.shape[1]; j++) {
                        t.t[i][j] = this.t[i].reduce((acc, val, k) => acc + (val * b.t[k][j]), 0);
                    }
                }
                return t;
            }
        }
        else {
            console.trace();
            throw "Dot Multiplication: Shape must be of size 2 (Matrix)..\n";
        }
    }
    padding(padding_height, padding_width, axis = [0, 1]) {
        if (axis == [0, 1]) {
            const t = new Tensor([2 * padding_height + this.shape[0], 2 * padding_width + this.shape[1], this.shape[3]], true);
            for (let i = 0; i < this.shape[0]; i++) {
                for (let j = 0; j < this.shape[1]; j++) {
                    for (let c = 0; c < this.shape[2]; c++) {
                        t.set([i + padding_height, j + padding_width, c], this.get([i, j, c]));
                    }
                }
            }
            return t;
        }
    }
    im2patches(patch_height, patch_width, filter_height, filter_width) {
        const cols = [];
        for (let r = 0; r < patch_height; r++) {
            for (let c = 0; c < patch_width; c++) {
                const v = [];
                for (let c_f_c = 0; c_f_c < this.shape[2]; c_f_c++) {
                    for (let c_f_h = 0; c_f_h < filter_height; c_f_h++) {
                        for (let c_f_w = 0; c_f_w < filter_width; c_f_w++) {
                            v.push(this.get([r + c_f_h, c + c_f_w, c_f_c]));
                        }
                    }
                }
                cols.push(v);
            }
        }
        return new Tensor(cols);
    }
    rotate180() {
        const t = this.copy(false);
        if (this.dim == 4) {
            this.iterate((n, i, j, k) => {
                t.t[n][this.shape[1] - 1 - i][this.shape[2] - 1 - j][k] = this.get([n, i, j, k]);
            });
        }
        return t;
    }
    rowVectors() {
        if (this.dim == 2) {
            return this.t.map((row) => new Tensor(row));
        }
    }
    argmax(index = -1, axis = 0) {
        if (this.dim == 1) {
            return this.t.reduce((acc, va, ind) => va > this.t[acc] ? ind : acc, 0);
        }
        else if (this.dim == 2) {
            if (axis == 0) {
                if (index < 0) {
                    return 0;
                }
                else {
                    return this.t[index].reduce((acc, va, ind) => va > this.t[index][acc] ? ind : acc, 0);
                }
            }
            else {
                if (index < 0) {
                    return 0;
                }
                else {
                    let maxIndex = 0;
                    for (let j = 0; j < this.shape[0]; j++) {
                        if (Math.abs(this.t[j][index]) > Math.abs(this.t[maxIndex][index])) {
                            maxIndex = j;
                        }
                    }
                    return maxIndex;
                }
            }
        }
    }
    reshape(shape) {
        if (this.dim == 1) {
            if (this.count() != shape.reduce((acc, n) => acc * n, 1)) {
                throw "Product of shape must be the same as size of vector!";
            }
            const t = new Tensor(shape, true);
            let [h, w, d] = shape;
            this.iterate((val, i) => {
                const r = Math.floor(i / (w * d));
                const c = Math.floor(i / (d) - (r * w));
                const g = Math.floor(i - (c * d) - (r * w * d));
                t.t[r][c][g] = val;
            });
            return t;
        }
    }
    sum(axis = -1, keepDims = false) {
        if (this.dim == 1) {
            return this.t.reduce((acc, val) => acc + val);
        }
        else if (this.dim == 2) {
            if (keepDims) {
                let t = this.copy();
                if (axis == 1) {
                    t.t.forEach((arr, i) => {
                        const sum = arr.reduce((acc, val) => acc + val, 0);
                        arr.forEach((val, j) => t.t[i][j] = sum);
                    });
                }
                else if (axis == 0) {
                    for (let j = 0; j < this.shape[1]; j++) {
                        let sum = 0;
                        for (let i = 0; i < this.shape[0]; i++) {
                            sum += this.t[i][j];
                        }
                        for (let i = 0; i < this.shape[0]; i++) {
                            t.t[i][j] = sum;
                        }
                    }
                }
                else if (axis == -1) {
                    const sum = t.t.reduce((acc, val) => {
                        acc += val.reduce((acc, val) => acc + val, 0);
                        return acc;
                    }, 0);
                    this.iterate((i, j) => {
                        t.t[i][j] = sum;
                    });
                }
                else if (axis >= 2) {
                    return this.copy();
                }
                return t;
            }
            else {
                if (axis == -1) {
                    return this.t.reduce((acc, val) => {
                        acc += val.reduce((acc, val) => acc + val, 0);
                        return acc;
                    }, 0);
                }
                else if (axis == 0) {
                    let t = new Tensor([1, this.shape[1]], true);
                    this.iterate((i, j) => {
                        t.t[0][j] = this.get([i, j]) + t.get([0, j]);
                    });
                    return t;
                }
                else if (axis == 1) {
                    let t = new Tensor([this.shape[0], 1], true);
                    this.t.forEach((arr, i) => {
                        t.t[i][0] = arr.reduce((acc, val) => acc + val, 0);
                    });
                    return t;
                }
                else if (axis == 2) {
                    return this.copy();
                }
                return 0;
            }
        }
    }
    mean(axis = -1, keep_dims = false) {
        if (this.dim == 1) {
            return this.sum() / this.count();
        }
        else if (this.dim == 2) {
            if (axis == -1) {
                return this.sum(-1, false) / this.count();
            }
            else if (axis == 0 || axis == 1) {
                return this.sum(axis, keep_dims).div(axis == 0 ? this.shape[0] : this.shape[1]);
            }
        }
    }
    repeat(axis = 0, times = 1) {
        if (this.dim == 2) {
            if (axis == 0) {
                const t = new Tensor([times, this.shape[1]], true);
                //t.t.fill(this.t[0])
                return t;
            }
        }
    }
    transpose() {
        let t = new Tensor([this.shape[1], this.shape[0]], true);
        this.iterate((i, j) => {
            t.t[j][i] = this.t[i][j];
        });
        return t;
    }
}
exports.default = Tensor;

"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const tensor_1 = __importDefault(require("./tensor"));
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

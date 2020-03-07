"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Vector = /** @class */ (function () {
    function Vector(defaultValue) {
        var _this = this;
        if (defaultValue === void 0) { defaultValue = new Float32Array(0); }
        this.size = function () {
            return _this.vector.length;
        };
        this.get = function (i) {
            return _this.vector[i];
        };
        this.set = function (i, n) {
            _this.vector[i] = n;
        };
        this.toString = function (vertical) {
            if (vertical === void 0) { vertical = true; }
            if (_this.vector.length == 0) {
                return "Vector: []";
            }
            else {
                if (vertical) {
                    return _this.vector.reduce(function (acc, i) {
                        acc += "    " + i + "\n";
                        return acc;
                    }, "Vector: [\n") + " ]";
                }
                else {
                    return _this.vector.reduce(function (acc, v) {
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
    Vector.fromBuffer = function (buff) {
        var v = new Vector(buff.length);
        for (var i = 0; i < v.size(); i++) {
            v.set(i, buff[i]);
        }
        return v;
    };
    Vector.toCategorical = function (index, size) {
        var v = new Vector(new Float32Array(size).fill(0));
        v.set(index, 1);
        return v;
    };
    Vector.prototype.toNumberArray = function () {
        return [].slice.call(this.vector);
    };
    Vector.prototype.populateRandom = function () {
        var _this = this;
        this.iterate(function (_, index) {
            _this.set(index, Math.random() * 2 - 1);
        });
    };
    Vector.prototype.iterate = function (func) {
        this.vector.forEach(function (value, index) {
            func(value, index);
        });
    };
    Vector.prototype.add = function (b) {
        var v = new Vector(this.size());
        if (b instanceof Vector) {
            if (b.size() != this.size())
                throw "Vectors to add aren't the same size..";
            this.iterate(function (val, i) {
                v.set(i, val + b.get(i));
            });
            return v;
        }
        else {
            var scalar_1 = b;
            this.iterate(function (val, i) {
                v.set(i, val + scalar_1);
            });
            return v;
        }
    };
    Vector.prototype.sub = function (b) {
        var v = new Vector(this.size());
        if (b instanceof Vector) {
            if (b.size() != this.size())
                throw "Vectors to subtract aren't the same size..";
            this.iterate(function (val, i) {
                v.set(i, val - b.get(i));
            });
            return v;
        }
        else {
            var scalar_2 = b;
            this.iterate(function (val, i) {
                v.set(i, val - scalar_2);
            });
            return v;
        }
    };
    Vector.prototype.mul = function (input) {
        var v = new Vector(this.size());
        if (input instanceof Vector) {
            if (input.size() != this.size()) {
                console.trace();
                throw "Vectors to multiply aren't the same size..";
            }
            this.iterate(function (val, i) {
                v.set(i, val * input.get(i));
            });
        }
        else {
            this.iterate(function (val, i) {
                v.set(i, val * input);
            });
        }
        return v;
    };
    Vector.prototype.div = function (scalar) {
        var v = new Vector(this.size());
        this.iterate(function (val, i) {
            v.set(i, val / scalar);
        });
        return v;
    };
    Vector.prototype.pow = function (scalar) {
        var v = new Vector(this.size());
        this.iterate(function (val, i) {
            v.set(i, Math.pow(val, scalar));
        });
        return v;
    };
    Vector.prototype.exp = function () {
        var v = new Vector(this.size());
        this.iterate(function (val, i) {
            v.set(i, Math.exp(val));
        });
        return v;
    };
    Vector.prototype.sum = function () {
        return this.vector.reduce(function (acc, val) { return acc + val; });
    };
    Vector.prototype.mean = function () {
        return this.sum() / this.size();
    };
    Vector.prototype.argmax = function () {
        var _this = this;
        return this.vector.reduce(function (acc, va, ind) { return va > _this.get(acc) ? ind : acc; }, 0);
    };
    return Vector;
}());
exports.default = Vector;

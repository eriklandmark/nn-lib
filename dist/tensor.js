"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var vector_1 = __importDefault(require("./vector"));
var matrix_1 = __importDefault(require("./matrix"));
var Tensor = /** @class */ (function () {
    function Tensor(v) {
        var _this = this;
        if (v === void 0) { v = []; }
        this.tensor = [];
        this.get = function (i, j, k) {
            return _this.tensor[i][j][k];
        };
        this.set = function (i, j, k, n) {
            _this.tensor[i][j][k] = n;
        };
        this.count = function () {
            return _this.dim().c * _this.dim().r * _this.dim().d;
        };
        this.toString = function (max_rows) {
            if (max_rows === void 0) { max_rows = 10; }
            if (_this.tensor.length == 0) {
                return "Tensor: 0x0x0 []";
            }
            else {
                var maxCharCount_1 = 0;
                _this.iterate(function (i, j, k) {
                    var val = _this.get(i, j, k).toString();
                    if (val.length > maxCharCount_1)
                        maxCharCount_1 = val.length;
                });
                maxCharCount_1 = Math.min(maxCharCount_1, 7);
                var string = "Tensor: " + _this.dim().r + "x" + _this.dim().c + "x" + _this.dim().d + " [\n";
                var _loop_1 = function (d) {
                    string += _this.tensor.slice(0, Math.min(max_rows, _this.tensor.length)).reduce(function (acc, i) {
                        acc += i.slice(0, Math.min(max_rows, i.length)).reduce(function (s, j) {
                            s += " ".repeat(Math.max(maxCharCount_1 - j[d].toString().length, 0));
                            s += j[d].toString().substr(0, maxCharCount_1) + " ";
                            return s;
                        }, "    ");
                        acc += i.length > max_rows ? " ... +" + (i.length - max_rows) + " elements\n" : "\n";
                        return acc;
                    }, "") + (_this.tensor.length > max_rows ?
                        "    ... +" + (_this.tensor.length - max_rows) + " rows \n" : "\n");
                };
                for (var d = 0; d < _this.dim().d; d++) {
                    _loop_1(d);
                }
                return string + "]";
            }
        };
        if (v.length > 0 && v[0][0] instanceof Float64Array) {
            this.tensor = v;
        }
        else {
            for (var i = 0; i < v.length; i++) {
                this.tensor.push([]);
                for (var j = 0; j < v[i].length; j++) {
                    this.tensor[i].push(Float64Array.from(v[i][j]));
                }
            }
        }
    }
    Tensor.prototype.dim = function () {
        return {
            r: this.tensor.length,
            c: this.tensor[0] ? this.tensor[0].length : 0,
            d: this.tensor[0][0] ? this.tensor[0][0].length : 0
        };
    };
    Tensor.prototype.shape = function () {
        return [this.dim().r, this.dim().c, this.dim().d];
    };
    Tensor.prototype.createEmptyArray = function (rows, columns, depth) {
        this.tensor = [];
        for (var i = 0; i < rows; i++) {
            this.tensor.push([]);
            for (var j = 0; j < columns; j++) {
                this.tensor[i].push(new Float64Array(depth).fill(0));
            }
        }
    };
    Tensor.fromJsonObject = function (obj) {
        return new Tensor(obj.map(function (row) {
            return row.map(function (col) {
                return Object.keys(col).map(function (item, index) { return col[index.toString()]; });
            });
        }));
    };
    Tensor.prototype.toNumberArray = function () {
        return this.tensor.map(function (array) { return array.map(function (floatArray) { return [].slice.call(floatArray); }); });
    };
    Tensor.prototype.iterate = function (func, channel_first) {
        if (channel_first === void 0) { channel_first = false; }
        if (channel_first) {
            for (var k = 0; k < this.dim().d; k++) {
                for (var i = 0; i < this.dim().r; i++) {
                    for (var j = 0; j < this.dim().c; j++) {
                        func(i, j, k);
                    }
                }
            }
        }
        else {
            for (var i = 0; i < this.dim().r; i++) {
                for (var j = 0; j < this.dim().c; j++) {
                    for (var k = 0; k < this.dim().d; k++) {
                        func(i, j, k);
                    }
                }
            }
        }
    };
    Tensor.prototype.copy = function (full) {
        var _this = this;
        if (full === void 0) { full = true; }
        var t = new Tensor();
        t.createEmptyArray(this.dim().r, this.dim().c, this.dim().d);
        if (full) {
            t.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k));
            });
        }
        return t;
    };
    Tensor.prototype.populateRandom = function () {
        var _this = this;
        this.iterate(function (i, j, k) {
            _this.set(i, j, k, Math.random() * 2 - 1);
        });
    };
    Tensor.prototype.empty = function () {
        return this.dim().c == 0 || this.dim().r == 0 || this.dim().d == 0;
    };
    Tensor.prototype.vectorize = function (channel_first) {
        var _this = this;
        if (channel_first === void 0) { channel_first = false; }
        var v = new vector_1.default(this.count());
        var index = 0;
        this.iterate(function (i, j, k) {
            v.set(index, _this.get(i, j, k));
            index += 1;
        }, channel_first);
        return v;
    };
    Tensor.prototype.div = function (val) {
        var _this = this;
        var t = this.copy(false);
        if (val instanceof Tensor) {
            if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                console.trace();
                throw "Tensor Division: Not the same dimension";
            }
            this.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k) / val.get(i, j, k));
            });
        }
        else {
            this.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k) / val);
            });
        }
        return t;
    };
    Tensor.prototype.mul = function (val) {
        var _this = this;
        var t = this.copy(false);
        if (val instanceof Tensor) {
            if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                console.trace();
                throw "Tensor Multiplication: Not the same dimension";
            }
            this.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k) * val.get(i, j, k));
            });
        }
        else {
            this.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k) * val);
            });
        }
        return t;
    };
    Tensor.prototype.sub = function (val) {
        var _this = this;
        var t = this.copy(false);
        if (val instanceof Tensor) {
            if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                console.trace();
                throw "Tensor Subtraction: Not the same dimension";
            }
            this.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k) - val.get(i, j, k));
            });
        }
        else {
            this.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k) - val);
            });
        }
        return t;
    };
    Tensor.prototype.add = function (val) {
        var _this = this;
        var t = this.copy(false);
        if (val instanceof Tensor) {
            if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                console.trace();
                throw "Tensor Subtraction: Not the same dimension";
            }
            this.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k) + val.get(i, j, k));
            });
        }
        else {
            this.iterate(function (i, j, k) {
                t.set(i, j, k, _this.get(i, j, k) + val);
            });
        }
        return t;
    };
    Tensor.prototype.padding = function (padding_height, padding_width) {
        var t = new Tensor();
        t.createEmptyArray(2 * padding_height + this.dim().r, 2 * padding_width + this.dim().c, this.dim().d);
        for (var i = 0; i < this.dim().r; i++) {
            for (var j = 0; j < this.dim().c; j++) {
                for (var c = 0; c < this.dim().d; c++) {
                    t.set(i + padding_height, j + padding_width, c, this.get(i, j, c));
                }
            }
        }
        return t;
    };
    Tensor.prototype.im2patches = function (patch_height, patch_width, filter_height, filter_width) {
        var cols = [];
        for (var r = 0; r < patch_height; r++) {
            for (var c = 0; c < patch_width; c++) {
                var v = [];
                for (var c_f_c = 0; c_f_c < this.dim().d; c_f_c++) {
                    for (var c_f_h = 0; c_f_h < filter_height; c_f_h++) {
                        for (var c_f_w = 0; c_f_w < filter_width; c_f_w++) {
                            v.push(this.get(r + c_f_h, c + c_f_w, c_f_c));
                        }
                    }
                }
                cols.push(new vector_1.default(v));
            }
        }
        return new matrix_1.default(cols);
    };
    Tensor.prototype.rotate180 = function () {
        var _this = this;
        var t = this.copy(false);
        this.iterate(function (i, j, k) {
            t.set(_this.dim().r - 1 - i, _this.dim().c - 1 - j, k, _this.get(i, j, k));
        });
        return t;
    };
    return Tensor;
}());
exports.default = Tensor;

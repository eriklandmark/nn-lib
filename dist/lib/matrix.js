"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var vector_1 = __importDefault(require("./vector"));
var node_worker_threads_pool_1 = require("node-worker-threads-pool");
var Matrix = /** @class */ (function () {
    function Matrix(defaultValue) {
        var _this = this;
        if (defaultValue === void 0) { defaultValue = []; }
        this.matrix = [];
        this.get = function (i, j) {
            return _this.matrix[i][j];
        };
        this.set = function (i, j, n) {
            _this.matrix[i][j] = n;
        };
        this.count = function (i, j, n) {
            return _this.dim().c * _this.dim().r;
        };
        this.toString = function (max_rows) {
            if (max_rows === void 0) { max_rows = 10; }
            if (_this.matrix.length == 0) {
                return "Matrix: 0x0 []";
            }
            else {
                var maxCharCount_1 = 0;
                _this.iterate(function (i, j) {
                    var val = _this.get(i, j).toString();
                    if (val.length > maxCharCount_1)
                        maxCharCount_1 = val.length;
                });
                maxCharCount_1 = Math.min(maxCharCount_1, 7);
                return _this.matrix.slice(0, Math.min(max_rows, _this.matrix.length)).reduce(function (acc, i) {
                    acc += i.slice(0, Math.min(max_rows, i.length)).reduce(function (s, i) {
                        s += " ".repeat(Math.max(maxCharCount_1 - i.toString().length, 0));
                        s += i.toString().substr(0, maxCharCount_1) + " ";
                        return s;
                    }, "    ");
                    acc += i.length > max_rows ? "  ... +" + (i.length - max_rows) + " elements\n" : "\n";
                    return acc;
                }, "Matrix: " + _this.dim().r + "x" + _this.dim().c + " [\n") + (_this.matrix.length > max_rows ?
                    "    ... +" + (_this.matrix.length - max_rows) + " rows \n]" : " ]");
            }
        };
        if (defaultValue.length > 0 && defaultValue[0] instanceof Float32Array) {
            this.matrix = defaultValue;
        }
        else if (defaultValue.length > 0 && defaultValue[0] instanceof vector_1.default) {
            var rows = defaultValue[0].size();
            var cols = defaultValue.length;
            this.createEmptyArray(rows, cols);
            this.iterate(function (i, j) {
                _this.set(i, j, defaultValue[j].get(i));
            });
        }
        else {
            for (var i = 0; i < defaultValue.length; i++) {
                this.matrix.push(Float32Array.from(defaultValue[i]));
            }
        }
    }
    Matrix.prototype.createEmptyArray = function (rows, columns) {
        for (var i = 0; i < rows; i++) {
            this.matrix.push(new Float32Array(columns).fill(0));
        }
    };
    Matrix.prototype.dim = function () {
        return { r: this.matrix.length, c: this.matrix[0] ? this.matrix[0].length : 0 };
    };
    Matrix.fromJsonObject = function (obj) {
        var m = new Matrix();
        m.createEmptyArray(obj.length, Object.keys(obj[0]).length);
        m.iterate(function (i, j) {
            m.set(i, j, obj[i][j.toString()]);
        });
        return m;
    };
    Matrix.prototype.toNumberArray = function () {
        return this.matrix.map(function (floatArray) { return [].slice.call(floatArray); });
    };
    Matrix.prototype.copy = function () {
        var _this = this;
        var m = new Matrix();
        m.createEmptyArray(this.dim().r, this.dim().c);
        m.iterate(function (i, j) {
            m.set(i, j, _this.get(i, j));
        });
        return m;
    };
    Matrix.prototype.iterate = function (func) {
        for (var i = 0; i < this.dim().r; i++) {
            for (var j = 0; j < this.dim().c; j++) {
                func(i, j);
            }
        }
    };
    Matrix.prototype.where = function (scalar) {
        var _this = this;
        this.iterate(function (i, j) {
            if (_this.get(i, j) == scalar) {
                return [i, j];
            }
        });
        return [-1, -1];
    };
    Matrix.prototype.populateRandom = function () {
        var _this = this;
        this.iterate(function (i, j) {
            _this.set(i, j, Math.random() * 2 - 1);
        });
    };
    Matrix.prototype.empty = function () {
        return this.dim().c == 0 || this.dim().r == 0;
    };
    Matrix.addGpu = function () {
        return function add(a, b) {
            //@ts-ignore
            return a + b;
        };
    };
    Matrix.subGpu = function () {
        return function sub(a, b) {
            //@ts-ignore
            return a - b;
        };
    };
    Matrix.multiplyGpu = function () {
        return function multiply(a, b) {
            //@ts-ignore
            return a * b;
        };
    };
    Matrix.mmGpu = function () {
        return function mm(a, b) {
            var sum = 0;
            for (var i = 0; i < a[0].length; i++) {
                sum += a[this.thread.y][i] * b[i][this.thread.x];
            }
            return sum;
        };
    };
    Matrix.prototype.mm = function (b, gpu) {
        var _this = this;
        if (gpu === void 0) { gpu = false; }
        if (b instanceof vector_1.default) {
            var v_1 = b;
            if (v_1.size() != this.dim().c) {
                console.trace();
                throw "Matrix Multiplication (Vector): Wrong dimension..";
            }
            var c = new vector_1.default(this.dim().r);
            for (var i = 0; i < this.dim().r; i++) {
                c.set(i, this.matrix[i].reduce(function (acc, val, k) { return acc + (val * v_1.get(k)); }, 0));
            }
            return c;
        }
        else {
            if (b.dim().r != this.dim().c) {
                console.trace();
                throw "Matrix Multiplication (Matrix): Wrong dimension..";
            }
            var m_1 = b;
            var c_1 = new Matrix();
            c_1.createEmptyArray(this.dim().r, m_1.dim().c);
            c_1.iterate(function (i, j) {
                c_1.set(i, j, _this.matrix[i].reduce(function (acc, val, k) { return acc + (val * m_1.get(k, j)); }, 0));
            });
            return c_1;
        }
    };
    Matrix.prototype.mmAsync = function (b) {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                return [2 /*return*/, new Promise(function (resolve, reject) { return __awaiter(_this, void 0, void 0, function () {
                        var c, i, c_2, pool_1;
                        var _this = this;
                        return __generator(this, function (_a) {
                            switch (_a.label) {
                                case 0:
                                    if (!(b instanceof vector_1.default)) return [3 /*break*/, 1];
                                    if (b.size() != this.dim().c) {
                                        reject("Matrix Multiplication (Vector): Wrong dimension..");
                                    }
                                    c = new vector_1.default(this.dim().r);
                                    for (i = 0; i < this.dim().r; i++) {
                                        c.set(i, this.matrix[i].reduce(function (acc, val, k) { return acc + (val * b.get(k)); }, 0));
                                    }
                                    resolve(c);
                                    return [3 /*break*/, 3];
                                case 1:
                                    if (!(b instanceof Matrix)) return [3 /*break*/, 3];
                                    if (b.dim().r != this.dim().c)
                                        reject("Matrix Multiplication (Matrix): Wrong dimension..");
                                    c_2 = new Matrix();
                                    c_2.createEmptyArray(this.dim().r, b.dim().c);
                                    pool_1 = new node_worker_threads_pool_1.StaticPool({
                                        size: Math.min(c_2.dim().r, 5),
                                        task: function (row) {
                                            var _a = this.workerData, matrix = _a.matrix, bMatrix = _a.bMatrix;
                                            var result = (new Float32Array(bMatrix[0].length)).map(function (_, col) {
                                                return matrix[row].reduce(function (acc, val, k) { return acc + (val * bMatrix[k][col]); }, 0);
                                            });
                                            return { i: row, v: result };
                                        },
                                        workerData: { matrix: this.matrix, bMatrix: b.matrix }
                                    });
                                    return [4 /*yield*/, (function () { return __awaiter(_this, void 0, void 0, function () {
                                            var row, _a, i, v, col;
                                            return __generator(this, function (_b) {
                                                switch (_b.label) {
                                                    case 0:
                                                        row = 0;
                                                        _b.label = 1;
                                                    case 1:
                                                        if (!(row < c_2.dim().r)) return [3 /*break*/, 4];
                                                        return [4 /*yield*/, pool_1.exec(row)];
                                                    case 2:
                                                        _a = _b.sent(), i = _a.i, v = _a.v;
                                                        for (col = 0; col < v.length; col++) {
                                                            c_2.set(i, col, v[col]);
                                                        }
                                                        _b.label = 3;
                                                    case 3:
                                                        row++;
                                                        return [3 /*break*/, 1];
                                                    case 4: return [2 /*return*/];
                                                }
                                            });
                                        }); })()];
                                case 2:
                                    _a.sent();
                                    resolve(c_2);
                                    _a.label = 3;
                                case 3: return [2 /*return*/];
                            }
                        });
                    }); })];
            });
        });
    };
    Matrix.prototype.add = function (b) {
        var m = this.copy();
        if (b instanceof Matrix) {
            if (b.dim().r != this.dim().r || b.dim().c != this.dim().c) {
                console.trace();
                throw "Matrix Addition: Not the same dimension";
            }
            this.iterate(function (i, j) {
                m.set(i, j, m.get(i, j) + b.get(i, j));
            });
            return m;
        }
        else {
            var scalar_1 = b;
            this.iterate(function (i, j) {
                m.set(i, j, m.get(i, j) + scalar_1);
            });
            return m;
        }
    };
    Matrix.prototype.sub = function (b) {
        var m = this.copy();
        if (b instanceof Matrix) {
            if (b.dim().r != m.dim().r || b.dim().c != m.dim().c) {
                console.trace();
                throw "Matrix Subtraction: Not the same dimension";
            }
            this.iterate(function (i, j) {
                m.set(i, j, m.get(i, j) - b.get(i, j));
            });
            return m;
        }
        else {
            var scalar_2 = b;
            this.iterate(function (i, j) {
                m.set(i, j, m.get(i, j) - scalar_2);
            });
            return m;
        }
    };
    Matrix.prototype.mul = function (b) {
        var m = this.copy();
        if (b instanceof Matrix) {
            if (b.dim().r != m.dim().r || b.dim().c != m.dim().c)
                throw "Matrix mult: Not the same dimension";
            this.iterate(function (i, j) {
                m.set(i, j, m.get(i, j) * b.get(i, j));
            });
        }
        else {
            this.iterate(function (i, j) {
                m.set(i, j, m.get(i, j) * b);
            });
        }
        return m;
    };
    Matrix.prototype.pow = function (scalar) {
        var m = this.copy();
        this.iterate(function (i, j) {
            m.set(i, j, Math.pow(m.get(i, j), scalar));
        });
        return m;
    };
    Matrix.prototype.exp = function () {
        var m = this.copy();
        this.iterate(function (i, j) {
            m.set(i, j, Math.exp(m.get(i, j)));
        });
        return m;
    };
    Matrix.prototype.log = function () {
        var m = this.copy();
        this.iterate(function (i, j) {
            m.set(i, j, Math.log(m.get(i, j)));
        });
        return m;
    };
    Matrix.prototype.sum = function (axis, keepDims) {
        var _this = this;
        if (axis === void 0) { axis = -1; }
        if (keepDims === void 0) { keepDims = false; }
        if (keepDims) {
            var m_2 = this.copy();
            if (axis == 1) {
                m_2.matrix.forEach(function (arr, i) {
                    var sum = arr.reduce(function (acc, val) { return acc + val; }, 0);
                    arr.forEach(function (val, j) { return m_2.set(i, j, sum); });
                });
            }
            else if (axis == 0) {
                var sum_1 = m_2.matrix.reduce(function (acc, val) {
                    acc += val.reduce(function (acc, val) { return acc + val; }, 0);
                    return acc;
                }, 0);
                this.iterate(function (i, j) {
                    m_2.set(i, j, sum_1);
                });
                return m_2;
            }
            else if (axis == 2) {
                return this.copy();
            }
            return m_2;
        }
        else {
            if (axis == -1) {
                return this.matrix.reduce(function (acc, val) {
                    acc += val.reduce(function (acc, val) { return acc + val; }, 0);
                    return acc;
                }, 0);
            }
            else if (axis == 0) {
                var m_3 = new Matrix();
                m_3.createEmptyArray(1, this.dim().c);
                this.iterate(function (i, j) {
                    m_3.set(0, j, _this.get(i, j) + m_3.get(0, j));
                });
                return m_3;
            }
            else if (axis == 1) {
                var m_4 = new Matrix();
                m_4.createEmptyArray(this.dim().r, 1);
                this.matrix.forEach(function (arr, i) {
                    var sum = arr.reduce(function (acc, val) { return acc + val; }, 0);
                    m_4.set(i, 0, sum);
                });
                return m_4;
            }
            else if (axis == 2) {
                return this.copy();
            }
            return 0;
        }
    };
    Matrix.prototype.div = function (scalar) {
        var m = this.copy();
        if (scalar instanceof Matrix) {
            this.iterate(function (i, j) {
                m.set(i, j, m.get(i, j) / scalar.get(i, j));
            });
        }
        else {
            this.iterate(function (i, j) {
                m.set(i, j, m.get(i, j) / scalar);
            });
        }
        return m;
    };
    Matrix.prototype.transpose = function () {
        var _this = this;
        var m = new Matrix();
        m.createEmptyArray(this.dim().c, this.dim().r);
        this.iterate(function (i, j) {
            m.set(j, i, _this.get(i, j));
        });
        return m;
    };
    Matrix.prototype.argmax = function (i, row) {
        var _this = this;
        if (i === void 0) { i = -1; }
        if (row === void 0) { row = true; }
        if (row) {
            if (i < 0) {
                return 0;
            }
            else {
                return this.matrix[i].reduce(function (acc, va, ind) { return va > _this.get(i, acc) ? ind : acc; }, 0);
            }
        }
        else {
            if (i < 0) {
                return 0;
            }
            else {
                var maxIndex = 0;
                for (var j = 0; j < this.dim().r; j++) {
                    if (Math.abs(this.get(j, i)) > Math.abs(this.get(maxIndex, i))) {
                        maxIndex = j;
                    }
                }
                return maxIndex;
            }
        }
    };
    Matrix.prototype.inv = function () {
        if (this.dim().c == 1 && this.dim().c == 1) {
            return new Matrix([[1 / this.get(0, 0)]]);
        }
        else if (this.dim().c == 2 && this.dim().c) {
            return new Matrix([
                [this.get(1, 1), -this.get(0, 1)],
                [-this.get(1, 0), this.get(0, 0)]
            ]).mul(1 / ((this.get(0, 0) * this.get(1, 1)) - (this.get(0, 1) * this.get(1, 0))));
        }
    };
    return Matrix;
}());
exports.default = Matrix;

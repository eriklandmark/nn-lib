"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = __importDefault(require("./matrix"));
var MatrixHelper = /** @class */ (function () {
    function MatrixHelper() {
    }
    MatrixHelper.row_reduction = function (matrix) {
        var m = matrix.copy();
        var h = 0;
        var k = 0;
        var swapArray = [];
        while (h < m.dim().r && k < m.dim().c) {
            var i_max = m.argmax(h, false);
            if (m.get(i_max) == 0) {
                k += 1;
            }
            else {
                var tempRow = m.matrix[h];
                m.matrix[h] = m.matrix[i_max];
                m.matrix[i_max] = tempRow;
                swapArray.push([h, i_max]);
                for (var i = h + 1; i < m.dim().r; i++) {
                    var f = m.get(i, k) / m.get(h, k);
                    m.set(i, k, 0);
                    for (var j = k + 1; j < m.dim().c; j++) {
                        m.set(i, j, m.get(i, j) - (m.get(i, j) * f));
                    }
                }
                h++;
                k++;
            }
        }
        for (var _i = 0, swapArray_1 = swapArray; _i < swapArray_1.length; _i++) {
            var _a = swapArray_1[_i], i = _a[0], j = _a[1];
            var tempRow = m.matrix[i];
            m.matrix[i] = m.matrix[j];
            m.matrix[j] = tempRow;
        }
        return m;
    };
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
    MatrixHelper.linear_least_squares = function (x, y) {
        var A = new matrix_1.default();
        A.createEmptyArray(x.size(), 2);
        A.matrix.forEach(function (val, index) {
            A.set(index, 0, 1);
            A.set(index, 1, x.get(index));
        });
        var VL = A.transpose().mm(A);
        var HL = A.transpose().mm(y);
        var xV = VL.inv().mm(HL);
        console.log(xV.toString());
        //const reducedVL = this.row_reduction(VL)
        //let zerosCount = new Vector(reducedVL.dim().r)
        //reducedVL.iterate((i,j) => {zerosCount.set(i, zerosCount.get(i) + (reducedVL.get(i,j) == 0? 1 : 0))})
    };
    return MatrixHelper;
}());
exports.default = MatrixHelper;

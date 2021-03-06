import Tensor from "../tensor";

export default class MatrixHelper {
    /*
    public static row_reduction(matrix: Tensor): Tensor {
        /*let m = matrix.copy();
        let h = 0
        let k = 0
        let swapArray = []

        while( h < m.dim().r && k < m.dim().c) {
            let i_max = m.argmax(h, false,);
            if (m.get(i_max) == 0) {
                k += 1
            } else {
                const tempRow = m.matrix[h];
                m.matrix[h] = m.matrix[i_max]
                m.matrix[i_max] = tempRow;
                swapArray.push([h, i_max])

                for (let i = h + 1; i < m.dim().r; i++) {
                    let f = m.get(i, k) / m.get(h, k)
                    m.set(i, k, 0)

                    for (let j = k + 1; j < m.dim().c; j++) {
                        m.set(i,j, m.get(i,j) - (m.get(i,j) * f))
                    }
                }

                h++
                k++
            }
        }

        for (let [i, j] of swapArray) {
            const tempRow = m.matrix[i];
            m.matrix[i] = m.matrix[j]
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
    }

    public static linear_least_squares(x: Vector, y: Vector) {
        let A = new Matrix()
        A.createEmptyArray(x.size(), 2)
        A.matrix.forEach((val: Float32Array, index) => {
            A.set(index,0, 1)
            A.set(index,1, x.get(index))
        })

        const VL: Matrix = <Matrix> A.transpose().mm(A)
        const HL: Vector = <Vector> A.transpose().mm(y)

        let xV = VL.inv()!.mm(HL)

        console.log(xV.toString())
        //const reducedVL = this.row_reduction(VL)
        //let zerosCount = new Vector(reducedVL.dim().r)
        //reducedVL.iterate((i,j) => {zerosCount.set(i, zerosCount.get(i) + (reducedVL.get(i,j) == 0? 1 : 0))})
    }

    public static linear_least_squares_pol(x: Vector, y: Vector) {
        let A = new Matrix()
        A.createEmptyArray(x.size(), 2)
        A.matrix.forEach((val: Float32Array, index) => {
            A.set(index,0, 1)
            A.set(index,1, x.get(index))
        })

        const VL: Matrix = <Matrix> A.transpose().mm(A)
        const HL: Vector = <Vector> A.transpose().mm(y)

        let xV = VL.inv()!.mm(HL)

        console.log(xV.toString())
        //const reducedVL = this.row_reduction(VL)
        //let zerosCount = new Vector(reducedVL.dim().r)
        //reducedVL.iterate((i,j) => {zerosCount.set(i, zerosCount.get(i) + (reducedVL.get(i,j) == 0? 1 : 0))})
    }*/
}
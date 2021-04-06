import Tensor from "../src/tensor";


console.log(new Tensor([[2, 2],[1, 2 ]]).inv())
console.log(new Tensor([[2, 2, 4],[1, 2 , 4],[1, 2, 19]]).inv())
const B = [
    [ 1, 2, -1],
    [ 2, 3, -1],
    [-2, 0, -3]
]

const C = [[1, 2, 3, 4], [7, 54, 42, 4], [135, 54, 51, 5], [64, 614, 64, 61]]

const t = new Tensor(C).inv()
t.print()





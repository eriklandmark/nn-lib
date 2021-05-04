import Tensor from "../src/tensor";


//console.log(new Tensor([[2, 2],[1, 2 ]]).inv())
//console.log(new Tensor([[2, 2, 4],[1, 2 , 4],[1, 2, 19]]).inv())
//const t = new Tensor([[1, 2, 3, 4], [7, 54, 42, 4], [135, 54, 51, 5], [64, 614, 64, 61]])


const A = new Tensor([
    [ -1, 1, -3],
    [ 2, 0, 7],
    [-1, -3, -7]
])

const b = new Tensor([2,-3,2])

console.time("1")
const t_1 = A.solve(b)
console.timeEnd("1")

console.time("2")
const t_2 = A.solve2(b)
console.timeEnd("2")

console.log(t_1.sub(t_2).norm())
console.log(A.cond())






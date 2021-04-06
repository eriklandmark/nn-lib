import Tensor from "../src/tensor";

const A = new Tensor([[1, 2],
                         [3, 4]])
const b = new Tensor([5,6]).transpose()

const x = A.solve(b)
x.print()

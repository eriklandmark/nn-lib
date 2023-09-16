import Tensor from "../src/tensor.ts";

const t_1 = new Tensor([[1], [2], [3]])
const t_2 = new Tensor([1, 2, 3])

t_1.print()
t_2.print()

console.log(t_2.equal(t_1))

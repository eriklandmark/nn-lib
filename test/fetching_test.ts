import Tensor from "../src/tensor.ts";

const t_1 = new Tensor([[1],[2],[3],[4]])

t_1.print()

console.log(t_1.get([2, 0]))

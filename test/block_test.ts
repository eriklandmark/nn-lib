import Tensor from "../src/tensor.ts";

const t_1 = new Tensor([[1,2], [3,4]])
const t_2 = new Tensor([[5, 6], [7, 8]])

t_1.print()
t_2.print()

t_1.concatenate(t_2, "v").print()


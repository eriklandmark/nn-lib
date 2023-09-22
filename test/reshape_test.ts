import Tensor from "../src/tensor.ts";

const t_1 = new Tensor([1, 2, 3, 1, 2, 3, 4,3])

t_1.print()
t_1.reshape([4,2]).print()

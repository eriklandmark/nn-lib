import Matrix from "./matrix";
import Vector from "./vector";

let a = new Matrix([
    [1, 2, 5],
    [3, 4, 6]]);

let v = new Vector([6,7]);
let b = new Matrix([
    [5, 6],
    [7, 8],
    [9, 10]]);



console.log(b.toString())

console.log(b.mm(a).toString())
console.log(b.div(2).toString())
console.log(b.transpose().toString())


import Matrix from "./lib/matrix";
import Vector from "./lib/vector";

let a = new Matrix([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]]);

let b = new Matrix([
    [1, 2, 3, 4],
    [7, 8, 1, 3],
    [9, 10, 5, 6]]);

let v = new Vector([6,7,0]);
let v2 = new Vector([8,9,8]);

let d = new Matrix([[8,1,1]])
let c = new Matrix([v2]);

console.log(d.mm(c).toString())

//console.log(a.mm(b).toString())
//console.log(b.transpose().mm(a.transpose()).toString())

/*console.log(b.toString())
console.log(b.add(3).toString())
console.log(b.mm(a).toString())
console.log(b.div(2).toString())
console.log(b.transpose().toString())
console.log(v.add(5).toString())*/
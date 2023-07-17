import Tensor from "../src/tensor";
import Helper from "../src/lib/helper";
import NC from "nodeaffinity"

console.log("Starting benchmarking...")
console.log(NC.setAffinity(2))

const TRIES = 50;

const size = 4000;
const dot_size = 400;
const a = new Tensor([size, size], true)
const b = new Tensor([size, size], true)
const c = new Tensor([dot_size, dot_size], true)
const d = new Tensor([dot_size, dot_size], true)
let populate_score = 0

const score_matrix = new Tensor([5, TRIES], true)
const trial_times = new Tensor([TRIES], true)

for (let i = 0; i < TRIES; i++) {
    trial_times.set([i], Helper.timeitSync(() => {
        populate_score = (Helper.timeitSync(() => {
            a.populateRandom(1337)
            b.populateRandom(1337)
            c.populateRandom(1337)
            d.populateRandom(1337)
        }, false) * 1000)
        score_matrix.set([0, i],Helper.timeitSync(() => {a.add(b)}, false) * 1000)
        score_matrix.set([1, i],Helper.timeitSync(() => {a.sub(b)}, false) * 1000)
        score_matrix.set([2, i],Helper.timeitSync(() => {a.div(b)}, false) * 1000)
        score_matrix.set([3, i],Helper.timeitSync(() => {a.exp()}, false) * 1000)
        score_matrix.set([4, i],Helper.timeitSync(() => {c.dot(d)}, false) * 1000)
    }, false))
}

const total_score = <number> score_matrix.sum() / TRIES
const total_time = trial_times.sum()

console.log("Total score:", total_score, "Total time:", total_time, "seconds")
score_matrix.print()


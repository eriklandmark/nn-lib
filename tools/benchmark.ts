import ProgressBar from "../src/lib/progress_bar.ts";
import Tensor from "../src/tensor.ts";
import Helper from "../src/lib/helper.ts";


console.log("Starting benchmarking...")


const bar = new ProgressBar(
    'Benchmarking: [' + '{bar}' + '] - {action}',
    7,
    {
        action: "--",
    }, 30)

bar.start()

const size = 4000;
const dot_size = 400;
const a = new Tensor([size, size], true)
const b = new Tensor([size, size], true)
const c = new Tensor([dot_size, dot_size], true)
const d = new Tensor([dot_size, dot_size], true)
let populate_score = 0
let add_score = 0
let sub_score = 0
let div_score = 0
let exp_score = 0
let dot_score = 0

Helper.timeit(() => {
    bar.increment({action: "populate"})
    populate_score = (2*(size**2) + 2*(dot_size**2)) / (Helper.timeitSync(() => {
        a.populateRandom()
        b.populateRandom()
        c.populateRandom()
        d.populateRandom()
    }, false) * 1000)
    bar.increment({action: "add"})
    add_score = (size**2) / (Helper.timeitSync(() => {a.add(b)}, false) * 1000)
    bar.increment({action: "subtract"})
    sub_score = (size**2) / (Helper.timeitSync(() => {a.sub(b)}, false) * 1000)
    bar.increment({action: "div"})
    div_score = (size**2) / (Helper.timeitSync(() => {a.div(b)}, false) * 1000)
    bar.increment({action: "exp"})
    exp_score = (size**2) / (Helper.timeitSync(() => {a.exp()}, false) * 1000)
    bar.increment({action: "dot"})
    dot_score = dot_size**3 / (Helper.timeitSync(() => {c.dot(d)}, false) * 1000)

}, false).then((seconds) => {
    bar.increment({action: "done"})
    bar.stop()
    const normal_score = add_score + sub_score + div_score + exp_score
    const total_score = Math.floor(populate_score + normal_score + dot_score)
    // console.log("Populate score: ", populate_score, "Algebraic score:", normal_score, "Dot score:", dot_score)
    console.log("Total score: ", total_score, "Total time:", seconds, "seconds")
})




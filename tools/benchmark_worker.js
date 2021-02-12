const Tensor = require("../dist/tensor").default;
const Helper = require("../dist/lib/helper").default;
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

process.on("message", async () => {
    try {
        populate_score = (2*(size**2) + 2*(dot_size**2)) / (Helper.timeitSync(() => {
            a.populateRandom()
            b.populateRandom()
            c.populateRandom()
            d.populateRandom()
        }, false) * 1000)
        //bar.increment({action: "add"})
        add_score = (size**2) / (Helper.timeitSync(() => {a.add(b)}, false) * 1000)
        //bar.increment({action: "subtract"})
        sub_score = (size**2) / (Helper.timeitSync(() => {a.sub(b)}, false) * 1000)
        //bar.increment({action: "div"})
        div_score = (size**2) / (Helper.timeitSync(() => {a.div(b)}, false) * 1000)
        //bar.increment({action: "exp"})
        exp_score = (size**2) / (Helper.timeitSync(() => {a.exp()}, false) * 1000)
        //bar.increment({action: "dot"})
        dot_score = dot_size**3 / (Helper.timeitSync(() => {c.dot(d)}, false) * 1000)
        const normal_score = add_score + sub_score + div_score + exp_score
        const total_score = Math.floor(populate_score + normal_score + dot_score)
        process.send(total_score);

    } catch (e) {
        process.send(0);
    }
})
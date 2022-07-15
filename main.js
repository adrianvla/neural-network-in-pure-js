var canvas = document.getElementById('cnv');
var ctx = canvas.getContext('2d', {
    alpha: false
});

ctx.imageSmoothingEnabled = false;
ctx.fillStyle = '#fff';
ctx.fillRect(0, 0, 500, 500);

ctx.msImageSmoothingEnabled = false;
ctx.mozImageSmoothingEnabled = false;
ctx.webkitImageSmoothingEnabled = false;
ctx.imageSmoothingEnabled = false;

function createLineElement(x, y, length, angle, colour, text, id) {
    var line = document.createElement("div");
    var styles = 'border: 1px solid ' + colour + '; ' +
        'width: ' + length + 'px; ' +
        'height: 2px; ' +
        '-moz-transform: rotate(' + angle + 'rad); ' +
        '-webkit-transform: rotate(' + angle + 'rad); ' +
        '-o-transform: rotate(' + angle + 'rad); ' +
        '-ms-transform: rotate(' + angle + 'rad); ' +
        'position: absolute; ' +
        'top: ' + y + 'px; ' +
        'left: ' + x + 'px; ' +
        'font-size: 20px;' +
        'display:flex;' +
        'justify-content:center;' +
        'align-items:center;' +
        'color:#fff;' +
        'background:' + colour + ';';
    line.setAttribute('style', styles);
    line.innerHTML = text;
    line.id = id;
    return line;
}

function createLine(x1, y1, x2, y2, colour, text, id) {
    var a = x1 - x2,
        b = y1 - y2,
        c = Math.sqrt(a * a + b * b);

    var sx = (x1 + x2) / 2,
        sy = (y1 + y2) / 2;

    var x = sx - c / 2,
        y = sy;

    var alpha = Math.PI - Math.atan2(-b, a);

    return createLineElement(x, y, c, alpha, colour, text, id);
}

$('.lines').append(createLine(10, 10, 50, 60, "#fff"));

var EPSILON = 0.0001;

class NN {
    constructor() {
        this.datasets = [];
    }
    async nd(size, type) {
        var ds = [];
        for (var i = 0; i < size; i++) {
            ds.push([Math.random() > 0.5 ? Math.random() * 2 : -Math.random() * 2, Math.random()]); // 0 in pos 0 is the bias here, we'll change it later when training       0 in pos 1 is the value of the neuron that ranges from 0 to 1 in R
        }



        this.datasets.push(ds);

        if (type == 0) { // start
            await this.endInit();
        }


        return this;
    }
    async endInit() {
        var that = this;
        for (var i = 0; i < this.datasets.length - 1; i++) {
            //console.log(that.datasets[i]);
            for (var j = 0; j < that.datasets[i].length; j++) {
                //var neuron =  that.datasets[i][j]
                for (var k = 0; k < that.datasets[i + 1].length; k++) {
                    that.datasets[i][j][k + 2] = Math.random() > 0.5 ? Math.random() : -Math.random();
                }
            }
        }


    }
    async calculate(input) {
        var that = this;
        for (var i = 0; i < input.length; i++) {
            that.datasets[that.datasets.length - 1][i][1] = input[i];
        }
        for (var i = that.datasets.length - 2; i > -1; i--) {
            //console.log(that.datasets[i]);
            for (var j = 0; j < that.datasets[i].length; j++) {
                //console.log(that.datasets[i][j]);
                var ws = 0;
                for (var k = 0; k < that.datasets[i][j].length - 2; k++) { // or -3
                    ws += that.datasets[i][j][k + 2] * that.datasets[i + 1][k][1];
                    //console.log(i+1)
                }
                that.datasets[i][j][1] = O(ws + that.datasets[i][j][0]);
            }
        }
    }
    async train(inputs, outputs) {
        var that = this;
        var vectornum = await this.save();/*
        var o = [];
        for(var u = 0;u<vectornum.length;u++) {
            o[u]=Math.random()*200 * (Math.random() > 0.5 ? 1 : -1);
        }*/
        var minim = await locMinFn(vectornum,this.cost, [inputs, outputs,this]);

        console.log(minim)


        return costs;
    }

    async nudge(u, v) {
        for (var i = 0; i < u.length; i++) {
            v[i] += u[i];
        }
        return v;
    }
    async cost(v, inputs, outputs,that) {
        await that.load(v);
        //console.log(v,inputs,outputs,that)
        var costs = 0;
        for (var j = 0; j < inputs.length; j++) {
            await that.calculate(inputs[j]);
            var cost = 0;
            for (var i = 0; i < that.datasets[0].length; i++) {
                cost += (that.datasets[0][i][1] - outputs[j][i]) ** 2;
            }
            costs += cost;
        }
        costs /= inputs.length;
        return costs;
    }
    async load(v) {
        var that = this;
        var index = 0;
        for (var i = 0; i < that.datasets.length - 1; i++) { //for weights and biases
            for (var j = 0; j < that.datasets[i].length; j++) {
                that.datasets[i][j][0] = v[index];
                index++;
                for (var k = 0; k < that.datasets[i][j].length - 2; k++) {
                    that.datasets[i][j][k + 2] = v[index];
                    index++;
                }
            }
        }

    }
    async save() {
        var that = this;
        var v = [];
        for (var i = 0; i < that.datasets.length - 1; i++) { //for weights and biases
            for (var j = 0; j < that.datasets[i].length; j++) {
                v.push(that.datasets[i][j][0]);
                for (var k = 0; k < that.datasets[i][j].length - 2; k++) {
                    v.push(that.datasets[i][j][k + 2]);
                }
            }
        }
        return v;
    }
}

function O(x) {
    return 1 / (1 + (Math.E ** (-x)));
}

function ddxO(x) {
    return dd(O(x));
}

function dd(x) {
    return x * (1 - x);
}

function ReLu(x) {
    return Math.max(0, x);
}
async function locMinFn(n_n, fn, args) {
    async function h(n_n, fn, args){
        if (!args) args = [];
        var turnables = 0;
        var goleft = false;
        var pval = await fn(turnables + n_n, ...args);
        turnables = EPSILON;
        if (pval < await fn(turnables + n_n, ...args)) {
            goleft = true;
        }
        pval = fn(n_n, ...args);
        var flag = true;
    
        while (flag) {
            if (!goleft)
                turnables += EPSILON;
            if (goleft)
                turnables -= EPSILON;
    
            if (pval < await fn(turnables+n_n, ...args)) {
                flag = false; //break out of while loop without break out of for loop
            }
            pval = await fn(turnables+n_n, ...args);
        }
        //n_n=await nudge(turnables,n_n);
    
        return turnables;
    }
    async function nudge(u, v) {
        for (var i = 0; i < u.length; i++) {
            v[i] += u[i];
        }
        return v;
    }
    var turnables = [];
    for(var i = 0;i<n_n.length;i++){
        turnables[i]=await h(n_n[i],async function(n){
            var arr = [];
            for(var u = 0;u<n_n.length;u++){
                arr[u]=0;
            }
            arr[i]=n;
            return await fn(arr,...args);
        },args);
    }
    //return await nudge(n_n,turnables); 
    return turnables;
}
async function a(o) {
    return (3 * (o[0] ** 4)) + (-2 * (o[0] ** 2)) + (o[1] ** 4) + o[1] + o[0] + 3;
}

async function represent(j) {
    $('.ds').text('');
    $('.lines').text('');

    for (var i = (j.length - 1); i > -1; i--) { //datasets
        var ds1 = $(`
        <div class="ds-1"></div>
        `)[0];
        for (var ii = 0; ii < j[i].length; ii++) {
            var u = j[i][ii][1];
            var nr = $(`
        <div class="neuron" style="border:hsl(calc(365 * ` + u + `),100%,50%) 7px solid">` + String(u).slice(0, 4) + `</div>
        `)[0];
            ds1.append(nr);
        }
        $('.ds').append(ds1);
    }
    for (var i = (j.length - 2); i > -1; i--) { //datasets, skip the first one because they don't have weights
        for (var ii = 0; ii < j[i].length; ii++) { //this neuron number
            var neuronnumber = ii; //j[i+1].length+ii;
            for (var n = (j.length - 1); n > i; n--) {
                neuronnumber += j[n].length;
            }
            for (var w = 0; w < j[i][ii].length - 2; w++) {
                var previousneuronnumber = w;
                for (var n = (j.length - 1); n > i + 1; n--) {
                    previousneuronnumber += j[n].length;
                }
                var u = j[i][ii][w + 2];
                var x1 = $($(".ds .neuron")[previousneuronnumber]).position().left + ($($(".ds .neuron")[previousneuronnumber]).width() / 2);
                var y1 = $($(".ds .neuron")[previousneuronnumber]).position().top + ($($(".ds .neuron")[previousneuronnumber]).height() / 2);
                var x2 = $($(".ds .neuron")[neuronnumber]).position().left + ($($(".ds .neuron")[previousneuronnumber]).width() / 2);
                var y2 = $($(".ds .neuron")[neuronnumber]).position().top + ($($(".ds .neuron")[previousneuronnumber]).height() / 2);
                var id = '_' + crypto.randomUUID();
                var line = createLine(x1, y1, x2, y2, "hsl(calc(365 * " + u + "),100%,50%)", `<span style="margin-left:` + String((w * 70) % (Math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2)))) + `px">` + String(u).slice(0, 4) + `</span>`, id);
                $(line).attr('val', u)
                $('.lines').append(line);




                $('#' + id).mouseenter(function () {
                    $('.hoverthing').text(String($(this).attr('val')).slice(0, 6));
                    $('.hoverthing').removeClass('dn');
                });
                $('#' + id).on('mousemove', function () {
                    $('.hoverthing').css('transform', 'translate(' + mouse[0] + 'px,' + mouse[1] + 'px)')
                });
                $('#' + id).mouseleave(function () {
                    $('.hoverthing').addClass('dn');
                });
            }
        }
    }
}
var mouse = [];
$(document).on("mousemove", function (event) {
    mouse[0] = event.pageX;
    mouse[1] = event.pageY;
});

var nn = new NN();
var randomset;
setInterval(function () {
    if (false) {

        nn = new NN();
        nn.nd(1, 2);
        nn.nd(2, 1);
        nn.nd(3, 0);
        randomset = [];
        for (var i = 0; i < 3; i++) {
            randomset.push(Math.random())
        }
        nn.calculate(randomset)
    }

    represent(nn.datasets);
}, 1000);

nn.nd(1, 2);
//nn.nd(2, 1);
nn.nd(2, 0);
randomset = [];
for (var i = 0; i < 2; i++) {
    randomset.push(Math.random())
}
nn.calculate(randomset)
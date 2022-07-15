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

class NN {
    constructor() {
        this.datasets = [];
    }
    async nd(size, type) {
        var ds = [];
        for (var i = 0; i < size; i++) {
            ds.push([Math.random() > 0.5 ? Math.random() * 2 : -Math.random() * 2, Math.random()]); // 0 in pos 0 is the bias here, we'll change it later when training       0 in pos 1 is the value of the neuron that ranges from 0 to 1 in R
        }

        if (type != 0) { //isn't start so that we can connect neurons in the n-1 dataset
            for (var i = 0; i < this.datasets[this.datasets.length - 1].length /*previous dataset*/ ; i++) {
                for (var ii = 0; ii < size; ii++) {
                    this.datasets[this.datasets.length - 1][i][ii + 2] = Math.random(); //we won't overwrite the bias in this neuron
                }
            }
        }


        this.datasets.push(ds);



        return this;
    }
    async calculate(input) {
        var that = this;
        for (var i = 0; i < input.length; i++) {
            that.datasets[0][i][1] = input[i];
        }
        for (var i = 1; i < that.datasets.length; i++) { //skip the start
            for (var j = 0; j < that.datasets[i].length; j++) {
                var previousWeight = [];
                var previousNV = [];


                for (var k = 0; k < that.datasets[i - 1].length; k++) {
                    previousWeight[k] = that.datasets[i - 1][k][j + 2];
                    previousNV[k] = that.datasets[i - 1][k][1];
                }
                var sum = 0;
                for (var k = 0; k < previousWeight.length; k++) {
                    sum += previousWeight[k] * previousNV[k];
                }
                that.datasets[i][j][1] = O(sum + that.datasets[i][j][0]);
            }
        }
        return this;
    }
}

function O(x) {
    return 1 / (1 + (Math.E ** (-x)))
    //return Math.max(0, x);
}

function represent(j) {
    $('.ds').text('');

    for (var i = 0; i < j.length; i++) { //datasets
        var ds1 = $(`
        <div class="ds-1"></div>
        `)[0];
        for (var ii = 0; ii < j[i].length; ii++) {
            var u = j[i][ii][1];
            var nr = $(`
        <div class="neuron" style="border:hsl(calc(255 * ` + u + `),100%,50%) 7px solid">` + String(u).slice(0, 4) + `</div>
        `)[0];
            ds1.append(nr);
        }
        $('.ds').append(ds1);
    }
}
var nn = new NN();
var randomset;
setInterval(function () {
    if (true) {

        nn = new NN();
        nn.nd(16, 0);
        nn.nd(8, 1);
        nn.nd(3, 1);
        nn.nd(5, 1);
        nn.nd(2, 2);
        randomset = [];
        for (var i = 0; i < 16; i++) {
            randomset.push(Math.random())
        }
        nn.calculate(randomset)
    }

    represent(nn.datasets);
}, 1000);

nn.nd(16, 0);
nn.nd(8, 1);
nn.nd(3, 1);
nn.nd(5, 1);
nn.nd(2, 2);

randomset = [];
for (var i = 0; i < 16; i++) {
    randomset.push(Math.random())
}
nn.calculate(randomset)
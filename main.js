//http-server . --cors -o
var mousePressed = false;
var lastX, lastY;
var canvas;
var model;
var coords = [] ;


function InitThis() {
    canvas = document.getElementById('myCanvas').getContext("2d");

    $('#myCanvas').mousedown(function (e) {
        mousePressed = true;
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas').mousemove(function (e) {
        if (mousePressed) {
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas').mouseup(function (e) {
        getFrame();
        mousePressed = false;
    });
	    $('#myCanvas').mouseleave(function (e) {
        mousePressed = false;
    });
    
}


function Draw(x, y, isDown) {
    if (isDown) {
        canvas.beginPath();
        r = 255; g = 255; b = 255; a = 255;
        // Select a fill style
        canvas.strokeStyle = "rgba(" + r + "," + g + "," + b + "," + (a / 255) + ")";
        canvas.lineWidth = "10";
        canvas.lineJoin = "round";
        canvas.moveTo(lastX, lastY);
        canvas.lineTo(x, y);
        canvas.closePath();
        canvas.stroke();
    }
    lastX = x; lastY = y;
}



function clearArea() {
    // Use the identity matrix while clearing the canvas
    canvas.setTransform(1, 0, 0, 1, 0, 0);
    canvas.clearRect(0, 0, canvas.canvas.width, canvas.canvas.height);
}


function getFrame()
{
    
  //  imgData = canvas.getImageData(0, 0, 140, 140);

    const imageData = canvas.getImageData(0, 0, 140, 140);
    //convert to tensor 
    var tfImg = tf.fromPixels(imageData, 1);
    var smalImg = tf.image.resizeBilinear(tfImg, [28, 28]);
    smalImg = tf.cast(smalImg, 'float32');
    var tensor = smalImg.expandDims(0);
    tensor = tensor.div(tf.scalar(255));

    var pred = model.predict(tensor).dataSync();
    var index = pred.indexOf(Math.max.apply(Math, pred));
    console.log(index);

}

// function preprocess(imgData)
// {
// return tf.tidy(()=>{
	  
//     var tensor = tf.fromPixels(imgData).toFloat()
//     var offset = tf.scalar(255.0);

//     // Normalize the image 
//     var normalized = tf.scalar(1.0).sub(tensor.div(offset));
//     var resized = tf.image.resizeBilinear(normalized, [28, 28])
//     var sliced   = resized.slice([0, 0, 1], [28, 28, 1])
//     var batched = sliced.expandDims(0)
//     return batched
// })
// }

async function loadModel() {
  model = await tf.loadModel('models/model.json');
  console.log('Model loaded');  
};


loadModel();




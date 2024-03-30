export function crop_bounding(imgString, pred) {
  /* if bounding box is available crop the image to only send what is within 1.25x the bounding box
    img: base64 encoded string
    preds: a prediction with attribute .bbox that contains dimensions of bounding box
    return: img
  */

  return new Promise((resolve, reject) => {
    let img = new Image();

    img.onload = function () {
      let canvas = document.createElement("canvas");
      let context = canvas.getContext("2d");
      console.log("prediction:", pred);
      if (pred !== undefined) {
        let x = Math.max(0, pred.bbox[0] - 0.125 * pred.bbox[2]);
        let y = Math.max(0, pred.bbox[1] - 0.125 * pred.bbox[3]);
        let width = pred.bbox[2] * 1.25;
        let height = pred.bbox[3] * 1.25;

        if (x + width > img.width) {
          width = img.width - x;
        }
        if (y + height > img.height) {
          height = img.height - y;
        }

        canvas.width = width;
        canvas.height = height;

        context.drawImage(img, x, y, width, height, 0, 0, width, height);

        resolve(canvas.toDataURL());
      } else {
        resolve(imgString);
      }
    };

    img.onerror = function () {
      reject(new Error("Image load failed"));
    };

    img.src = imgString;
  });
}

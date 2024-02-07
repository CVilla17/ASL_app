import React, { Component } from "react";
import { useCallback, useRef, useState } from "react"; // import useCallback
import Webcam from "react-webcam";

const CustomWebcam = () => {
  const webcamRef = useRef(null); // create a webcam reference
  const [imgSrc, setImgSrc] = useState(null); // initialize it
  const [prediction, setPred] = useState(null);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);
  }, [webcamRef]);

  const retake = () => {
    setImgSrc(null);
    setPred(null);
  };

  const setImageAction = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    const response = await fetch(imgSrc);
    const blob = await response.blob();
    const file = new File([blob], "unknown_sign.jpg");
    formData.append("file", file);

    // console.log(imgSrc.pictureAsFile);

    for (var key of formData.entries()) {
      console.log(key[0] + ", " + key[1]);
    }

    const data = await fetch("http://localhost:8000/predict/image", {
      method: "post",
      headers: {},
      type: "image/jpeg",
      accept: "application/json",
      body: formData,
    });
    const uploadedImage = await data.json();
    if (uploadedImage) {
      console.log("Successfully uploaded image");
      console.log("guess", uploadedImage[0].predicted);
      console.log("guess", uploadedImage[0].confidence);
      setPred({
        prediction: uploadedImage[0].predicted,
        confidence: (uploadedImage[0].confidence * 100).toFixed(2),
      });
    } else {
      console.log("Error Found");
    }
  };

  return (
    <div className="container">
      {imgSrc ? (
        <img src={imgSrc} alt="webcam" />
      ) : (
        <Webcam
          height={600}
          width={600}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
        />
      )}
      <div className="btn-container">
        {imgSrc ? (
          <div>
            <button onClick={retake}>Retake photo</button>
            <form onSubmit={setImageAction}>
              <br />
              <br />
              <button type="submit" name="upload">
                Upload
              </button>
            </form>
          </div>
        ) : (
          <button onClick={capture}>Capture photo</button>
        )}
      </div>
      <div>
        {prediction ? (
          <div>
            {" "}
            The model thinks you signed {prediction.prediction} with{" "}
            {prediction.confidence}% confidence{" "}
          </div>
        ) : (
          <div></div>
        )}
      </div>
    </div>
  );
};

export default CustomWebcam;

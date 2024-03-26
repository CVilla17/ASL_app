import React, { Component } from "react";
import { useCallback, useRef, useState, useEffect } from "react"; // import useCallback
import Webcam from "react-webcam";
import * as handTrack from "handtrackjs";

const CustomWebcam = () => {
  const webcamRef = useRef(null); // create a webcam reference
  const [imgSrc, setImgSrc] = useState(null); // initialize it
  const [prediction, setPred] = useState(null);
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    let isCancelled = false;

    const loadModel = async () => {
      const modelParams = {
        flipHorizontal: false, // flip e.g for video
        imageScaleFactor: 0.9, // reduce input image size for gains in speed.
        maxNumBoxes: 2, // maximum number of boxes to detect
        iouThreshold: 0.8, // ioU threshold for non-max suppression
        scoreThreshold: 0.9, // confidence threshold for predictions.
      };
      const model = await handTrack.load(modelParams);
      if (!isCancelled) {
        setModel(model);
      }
    };

    loadModel();

    return () => {
      isCancelled = true;
    };
  }, []);

  const detect = useCallback(async () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      const prediction = await model.detect(video);
      setPredictions(prediction);
    }
  }, [model]);

  useEffect(() => {
    if (model !== null) {
      const interval = setInterval(() => {
        detect();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [model, detect]);

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
        <div style={{ position: "relative" }}>
          <Webcam
            height={600}
            width={600}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              zIndex: -1,
            }}
          />
          {predictions.map((prediction, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                top: prediction.bbox[1],
                left: prediction.bbox[0],
                height: prediction.bbox[3],
                width: prediction.bbox[2],
                borderWidth: 2,
                borderColor: "red",
                borderStyle: "solid",
              }}
            ></div>
          ))}
        </div>
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

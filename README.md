# ASL Alphabet tutor

This (currently in progress) project aims to provide an interactive learning experience for those interested in learning the ASL alphabet!
The web app combines a simple, randomized, flashcard style learning. To detect proper handsigns, the app
uses a machine learning model trained on a [dataset created by David Lee](https://public.roboflow.com/object-detection/american-sign-language-letters/1). The model is an extension of resnet50 from the pytorch library. Foundations of the models code come from
the github repository by [MLWhiz](https://github.com/MLWhiz/data_science_blogs/tree/master/compvisblog). Additionally, hand tracking from [Handtrack.js](https://github.com/victordibia/handtrack.js/) was used to aid in better detection and prediction of signs.

## Instructions for use:

Install requirements with the following command:

```

pip install -r requirement.txt

```

From the outermost folder, run the server. I use the following command:

```

python -m server

```

With the server running, navigate to the asl-alphabet-tutor folder and you can launch the front end with:

```

npm start

```

### Future updates

A learning environment that functions according to a Modified-Leitner algorithm to help aid in learning

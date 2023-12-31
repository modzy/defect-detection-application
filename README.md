# Modzy-powered Defect Detection App

![Modzy Logo](./imgs/banner.png)

<div align="center">

<p float="center">
    <img src="./imgs/dashboard.gif">
</p>

**This repository provides an example implementation of a Flask application integrated with computer vision models & Modzy edge for automated defect detection.**

![GitHub contributors](https://img.shields.io/github/contributors/modzy/defect-detection-application?logo=GitHub&style=flat)
![GitHub last commit](https://img.shields.io/github/last-commit/modzy/defect-detection-application?logo=GitHub&style=flat)
![GitHub issues](https://img.shields.io/github/issues-raw/modzy/defect-detection-application?logo=github&style=flat)
![GitHub](https://img.shields.io/github/license/modzy/defect-detection-application?logo=apache&style=flat)

<h3 align="center">
  <a href="https://docs.modzy.com/docs/edge">Modzy Edge Documentation</a>
</div>


## Overview

[Modzy edge](https://docs.modzy.com/docs/edge) provides the ability to deploy ML models to any remote device and integrate them into your custom applications. Example applications might include computer vision models running on camera devices in a manufacturing facility for worker safety, air quality models connected to sensors for air quality prediction, audio enhancement models for telecom applications, or countless other edge AI use cases. 

This repository includes an example implementation of a defect detection app that uses custom-trained YOLO models for detecting part defects on 3D-printed spur gears. Because the models are highly bespoke, the objective of this template is to simply provide a *framework* for creating custom applications using Python and Modzy edge APIs. As a result, it is expected that some of the model-specific details (e.g., model identifier, version, data source, etc.) will need to modified for your application.

*NOTE: This app was originally built to run on an NVIDIA Jetson Nano device, so some of the code may use device-specific developer packages, but this framework can be modified for other device and architecture types.*

## Technologies Used
Below is a list of the key technologies used to create this application. These technologies can be swapped out as needed, but doing so will require additional customization.

* Web app framework: [flask](https://github.com/pallets/flask)
* Web app containerization: [Docker](https://www.docker.com/)
* Video streaming: [gstreamer](https://github.com/GStreamer/gstreamer)
* Video stream processing: [opencv](https://github.com/opencv/opencv)
* Computer vision inference: [Modzy Edge](https://docs.modzy.com/docs/edge)

## Getting Started

This section outlines the set of prerequisites you will need before running and interacting with the defect detection Flask app:

1. Python environment (v3.7 or greater supported) 
2. Docker installation (we highly recommmend containerizing your app for easier repeatability and scale)
2. A running instance of [Modzy core](https://docs.modzy.com/docs/connect-edge-device)
3. Your model container(s) downloaded from your Modzy enterprise account

*NOTE: Modzy core will orchestrate the serving of your model(s) and expose them via Modzy's edge APIs. You may run your Flask app either on the same device or on a separate device as the device running Modzy core. The only difference in the Flask app code will be how you instantiate the Modzy edge client (via localhost if on same device or via IP address if on difference device). The following set of instructions does not distinguish between multiple device environments.*

With these prerequisites met, we can prepare our environment for local development and testing. First, clone this repository into your environment and navigate to the directory with the Flask app code:

```bash
git clone https://github.com/modzy/defect-detection-application.git
cd defect-detection-application/flask-app/
```

Next, create a virtual environment (virtualenv, venv, conda all work) and activate it.

```bash
python3 -m venv .env
```

_Linux or Mac OS_
```bash
source .env/bin/activate
```

_Windows_
```cmd
.\.env\Scripts\activate
```

Now, install the packages needed to run the Flask app:

```bash
pip install -r requirements.txt
```

With these Python packages installed, your environment should be set up and ready to run the Flask app. 

## Usage

### Running Modzy Core

As listed in the [Getting Started](./README.md#getting-started) section, you'll need to have an instance of Modzy core installed on your device that will capture a video stream and feed it through your model(s) for predictions. Assuming that you've installed Modzy Core and have waited for all model containers to download, you can run Modzy core in server mode by running the following command:

```bash
./modzy-core server --resume
```

*NOTE: Modzy Core needs to run in it's own terminal, so you'll need to open a second terminal on the same device to run the corresponding flask app the uses Modzy Core for inference.*

For more detailed Modzy Core installation instructions, please see our [documentation](https://docs.modzy.com/docs/edge).
 
### Running Flask App

Before running the application, there are a few changes you will need to make in the code itself.

First, navigate to [configuration section](./flask-app/app.py#L16-#L28) of the Flask app code and make any changes specific to your model and/or application. At a minimum, you will need to edit the `MODEL_ID`, `MODEL_VERSION`, and `NAMES` variables to represent your model's specific config and class labels.  

Next, edit the [edge client instantiation](./flask-app/app.py#L117) based on the location your Flask app is running relative to the device running Modzy core. If you are running this app on the same device, you will keep the host as "localhost". Otherwise, insert the IP address of your device running Modzy core.

*NOTE: If you are using this repository as a template for your use case, make any additional changes as you see fit before running your Flask app.*

With these changes made, you can run the Flask app code:

```bash
python3 app.py
```

If successful, you should see the following log lines in your terminal:

```
 * Serving Flask app 'stream_flask_new' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://<device-IP-address>:8000/ (Press CTRL+C to quit)
```

### Containerize Flask App

Now that you have verified your app can run locally in your Python environment, we *highly* recommend containerizing your app. Doing so will create an immutable asset that can be downloaded and run anywhere, seamlessly scaled, and repeatedly run without having to set up a clean Python environment.

To build a container out of your Flask app, simply run the following command in your terminal:

```bash
docker build -t defect-detection-app .
```

You will notice in the [Dockerfile](./Dockerfile), we set an environment variable for the port on which the Flask app is exposed (`ENV PORT=8000`). The default is 8000, but you may customize this at runtime. 

Run your container with this command:

```bash
docker run --rm -it -e PORT=8000 -p 8000:8000 defect-detection-app
```

You should see the same set of logs you saw when running this app locally. Open a browser and navigate to `http://127.0.0.1:8000` to check it out!

### Customization
To customize this template for your own purposes, here are some helpful tips:
 * Changes to the UI can be made in [flask-app/templates/app.html](./flask-app/templates/app.html) and [flask-app/static/css/style.css](./flask-app/static/css/style.css).
 * Changes to the video stream processing and AI inference processing can be made in [flask-app/app.py](./flask-app/app.py). The `app.py` script also includes comments with the word "EDITABLE" to highly sections of the codebase will likely need to change if using a different streaming protocol besides gstreamer, or for using this template with a different computer vision model. 
 * Changes to the container structure, base container image, environment variables or other container-related edits can be made directly in the [Dockerfile](./Dockerfile)

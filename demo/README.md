This folder holds our HF demo for the message decoding model. You can run this demo locally or deploy it to Spaces.

The demo has a leaner set of requirements than the rest of the project. Spaces requires that the dependecies are defined by a `requirements.txt` file, so you'll see that here.

Set up demo virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the demo:
```
CUDA_VISIBLE_DEVICES=0 gradio demo.py
```


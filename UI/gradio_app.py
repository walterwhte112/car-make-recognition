import gradio as gr
from utils.predict import predict_image

def gradio_predict(img):
    img.save("temp.jpg")
    return predict_image("temp.jpg")

gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Car Make Recognition",
    description="Upload a car image and the model will predict its make."
).launch()

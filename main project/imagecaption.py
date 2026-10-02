"""Image captioning module: pretrained BLIP (downloaded on first use)."""
from transformers import pipeline
def load_captioner():
    return pipeline('image-to-text',model='Salesforce/blip-image-captioning-base')
def generate_caption(image,captioner):
    result=captioner(image)
    return result[0]['generated_text']

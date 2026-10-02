"""LSTM inference with the same input template used during training."""
import json,re
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
BASE = Path(__file__).resolve().parent / 'models'
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()
def load_classifier():
    required = ['lstm_model.keras','tokenizer.json','labels.json','config.json']
    missing = [f for f in required if not (BASE/f).exists()]
    if missing: raise FileNotFoundError('Missing trained artifacts: '+', '.join(missing)+'; run the export cell in LSTM_Training.ipynb')
    model=load_model(BASE/'lstm_model.keras')
    tokenizer=tokenizer_from_json((BASE/'tokenizer.json').read_text(encoding='utf-8'))
    labels=json.loads((BASE/'labels.json').read_text(encoding='utf-8'))
    config=json.loads((BASE/'config.json').read_text(encoding='utf-8'))
    return model,tokenizer,labels,config
def classify(text, image_description, artifacts):
    model,tokenizer,labels,config=artifacts
    combined='query '+clean_text(text)+' image descriptions '+clean_text(image_description)
    seq=tokenizer.texts_to_sequences([combined])
    padded=pad_sequences(seq,maxlen=config['max_len'],padding='post',truncating='post')
    scores=model.predict(padded,verbose=0)[0]
    idx=int(np.argmax(scores))
    return labels[idx],float(scores[idx])

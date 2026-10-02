# Cellula NLP Submission

This project contains the NLP internship tasks developed for Cellula Technologies, including RNN and LSTM text classification models, a Streamlit demo application, image captioning integration, and quantization research.

## Folder Structure

- `notebooks/RNN/`
  - RNN training and evaluation notebook.
  - RNN results PDF.

- `notebooks/LSTM/`
  - LSTM training, evaluation, and experimentation notebook.
  - LSTM results PDF.

- `main project/`
  - Streamlit application.
  - LSTM inference module.
  - BLIP image captioning module.
  - CSV prediction database.
  - Model artifacts and configuration files.
  - Python requirements.

- `Task0_Quantization/`
  - Quantization research notebook.
  - Memory comparison experiment.
  - Toy INT8 quantization implementation.

- `cellula_3000_AUGMENTED_PROVISIONAL.csv`
  - Dataset used for the final training experiments.

## Model Results

### RNN

Validation results:

- Accuracy: ~0.95
- Macro F1: 0.9219
- Weighted F1: 0.9439

### LSTM

Validation results:

- Accuracy: ~0.97
- Macro F1: 0.9398
- Weighted F1: 0.9619

The LSTM model was selected for deployment in the Streamlit application.

## Streamlit Application

The application provides three main functions:

1. **Classify text**
   - Accepts user text.
   - Applies the same preprocessing and input template used during training.
   - Returns the predicted category and model confidence.

2. **Classify image**
   - Accepts an uploaded image.
   - Uses BLIP to generate an image caption.
   - Sends the generated caption, optionally together with user text, to the LSTM classifier.

3. **View CSV database**
   - Stores and displays previous predictions.
   - Records input type, text, generated image caption, predicted class, confidence, and timestamp.
   - Allows the database to be downloaded as CSV.

## Running the Application

From the project root:

```bash
cd "main project"
pip install -r requirements.txt
streamlit run app.py
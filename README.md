# AI Photo Album

A web-based photo gallery that generates captions for uploaded images using the [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) model.

## Setup

```bash
pip install flask torch transformers pillow
python app.py
```

Then open http://localhost:5001

## Features

- Upload images and get AI-generated captions automatically
- Browse uploaded images in a gallery grid
- Click any image to view it full-size with its caption
- Persistent storage via JSON file

## Tech Stack

- **Backend:** Python, Flask
- **ML Model:** Salesforce BLIP (image captioning)
- **Frontend:** HTML, Tailwind CSS
- **Storage:** JSON file (gallery.json)

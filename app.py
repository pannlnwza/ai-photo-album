from flask import Flask, request, render_template, jsonify
import os
import json
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
GALLERY_FILE = 'gallery.json'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and processor
processor = None
model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_gallery():
    if os.path.exists(GALLERY_FILE):
        with open(GALLERY_FILE, 'r') as f:
            return json.load(f)
    return {'albums': []}


def save_gallery(data):
    with open(GALLERY_FILE, 'w') as f:
        json.dump(data, f)


def load_model():
    global processor, model
    try:
        logger.info("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        processor = None
        model = None
        raise


def generate_caption(image_path):
    global model, processor
    try:
        if model is None or processor is None:
            load_model()
            if model is None or processor is None:
                return "Error: Model not loaded properly"

        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)

        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption if caption else "No caption generated"
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        return f"Error: {str(e)}"


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            with Image.open(image_path) as img:
                img_format = img.format.lower()
            return f"data:image/{img_format};base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/albums', methods=['GET'])
def get_albums():
    data = load_gallery()
    return jsonify(data['albums'])


@app.route('/albums', methods=['POST'])
def create_album():
    body = request.get_json()
    name = body.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Album name is required'}), 400

    data = load_gallery()
    album = {
        'id': uuid.uuid4().hex[:8],
        'name': name,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'images': [],
    }
    data['albums'].insert(0, album)
    save_gallery(data)
    return jsonify(album)


@app.route('/albums/<string:album_id>', methods=['DELETE'])
def delete_album(album_id):
    data = load_gallery()
    data['albums'] = [a for a in data['albums'] if a['id'] != album_id]
    save_gallery(data)
    return jsonify({'success': True})


@app.route('/albums/<string:album_id>/upload', methods=['POST'])
def upload_file(album_id):
    data = load_gallery()
    album = next((a for a in data['albums'] if a['id'] == album_id), None)
    if not album:
        return jsonify({'error': 'Album not found'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            caption = generate_caption(filepath)
            image_base64 = image_to_base64(filepath)

            os.remove(filepath)

            entry = {
                'id': uuid.uuid4().hex[:8],
                'filename': file.filename,
                'caption': caption,
                'image': image_base64,
                'uploaded_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            }
            album['images'].insert(0, entry)
            save_gallery(data)

            return jsonify({'success': True, **entry})
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400


@app.route('/albums/<string:album_id>/images/<string:image_id>', methods=['DELETE'])
def delete_image(album_id, image_id):
    data = load_gallery()
    album = next((a for a in data['albums'] if a['id'] == album_id), None)
    if not album:
        return jsonify({'error': 'Album not found'}), 404

    album['images'] = [img for img in album['images'] if img['id'] != image_id]
    save_gallery(data)
    return jsonify({'success': True})


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'processor_loaded': processor is not None,
    })


if __name__ == '__main__':
    try:
        load_model()
        print("Starting Flask application...")
        print("Model loaded successfully!")
        print("Open http://localhost:5001 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        print("pip install flask torch transformers pillow")

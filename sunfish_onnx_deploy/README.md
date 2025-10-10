# 🐟 SunFish ONNX Deployment Package

Complete ONNX package for deploying SunFish diffusion model.

## 📦 Contents

- `sunfish_model.onnx` - ONNX model (107MB)
- `token_embeddings.npy` - Token embeddings for text conversion
- `config.json` - Model configuration
- `inference.py` - Complete inference script
- `requirements.txt` - Dependencies

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate text
python inference.py --samples 3 --length 128 --steps 20
```

## 📊 Model Info

- **Architecture**: Diffusion Language Model
- **Parameters**: ~42M
- **Embedding Dim**: 512
- **Vocab Size**: 8192
- **Timesteps**: 500

## 🔧 API Usage

```python
from inference import SunFishONNX

# Load model
model = SunFishONNX()

# Generate text
texts = model.generate(num_samples=3, seq_len=128, num_steps=20)

for text in texts:
    print(text)
```

## 🌐 Platform Deployment

### ONNX Runtime (CPU)
```python
import onnxruntime as ort
session = ort.InferenceSession("sunfish_model.onnx")
```

### ONNX Runtime (GPU)
```python
session = ort.InferenceSession(
    "sunfish_model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### Web Deployment (ONNX.js)
```javascript
const session = await ort.InferenceSession.create('sunfish_model.onnx');
```

## ⚡ Performance

- **CPU**: ~2-3 seconds per sample
- **GPU**: ~0.5-1 second per sample
- **Batch inference**: Supported via batch_size parameter

## 🔌 Integration Example

```python
# Flask API
from flask import Flask, request, jsonify
from inference import SunFishONNX

app = Flask(__name__)
model = SunFishONNX()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    texts = model.generate(
        num_samples=data.get('samples', 1),
        seq_len=data.get('length', 128),
        num_steps=data.get('steps', 20)
    )
    return jsonify({'texts': texts})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

## 📝 Notes

- First run downloads GPT-2 tokenizer (~500MB)
- Supports batch inference for multiple samples
- Pure ONNX - no PyTorch dependency at inference time
- Cross-platform: Linux, Windows, macOS, Web

## 🎯 Deployment Checklist

- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Test inference: `python inference.py`
- [ ] Verify output quality
- [ ] Integrate into your platform
- [ ] (Optional) Optimize with ONNX Runtime GPU

---

**Ready for production deployment!** 🚀

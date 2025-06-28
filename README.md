# 🧠 Brain Tumor Detection - Backend API

A powerful Django REST API for brain tumor detection and analysis using deep learning models. Features both classification and segmentation capabilities with state-of-the-art AI models.

## 🚀 Features

- **AI-Powered Tumor Classification**: EfficientNetB7-based model for tumor type detection
- **Tumor Segmentation**: ResUNet architecture for precise tumor boundary detection
- **RESTful API**: Clean, documented API endpoints
- **User Authentication**: JWT-based authentication system
- **File Upload**: Secure MRI scan upload and processing
- **Real-time Analysis**: Fast inference with optimized models
- **Comprehensive Results**: Detailed analysis with confidence scores and measurements
- **CORS Support**: Cross-origin resource sharing for frontend integration

## 🛠️ Tech Stack

- **Framework**: Django 5.2 with Django REST Framework
- **AI/ML**: TensorFlow 2.13, Keras
- **Image Processing**: OpenCV, Pillow
- **Database**: SQLite (development), PostgreSQL (production ready)
- **Authentication**: Django REST Framework Token Authentication
- **CORS**: django-cors-headers
- **API Documentation**: Built-in DRF browsable API

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   **Windows:**
   ```bash
   .\venv\Scripts\Activate.ps1
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install django-cors-headers
   ```

5. **Navigate to Django project**
   ```bash
   cd tumor_detection
   ```

6. **Run database migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

7. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

8. **Start the development server**
   ```bash
   python manage.py runserver
   ```

9. **Access the API**
   Navigate to [http://127.0.0.1:8000/api/](http://127.0.0.1:8000/api/)

## 🏗️ Project Structure

```
Backend/
├── requirements.txt              # Python dependencies
├── tumor_detection/              # Main Django project
│   ├── manage.py                 # Django management script
│   ├── tumor_detection/          # Project settings
│   │   ├── settings.py           # Django settings
│   │   ├── urls.py               # Main URL configuration
│   │   ├── wsgi.py               # WSGI configuration
│   │   └── asgi.py               # ASGI configuration
│   ├── api/                      # Main API app
│   │   ├── models.py             # Database models
│   │   ├── views.py              # API views and logic
│   │   ├── serializers.py        # Data serializers
│   │   ├── urls.py               # API URL patterns
│   │   ├── admin.py              # Django admin configuration
│   │   └── migrations/           # Database migrations
│   ├── training/                 # Model training scripts
│   │   ├── train.py              # Training script
│   │   ├── model.py              # Model architecture
│   │   ├── data.py               # Data preprocessing
│   │   ├── config.py             # Training configuration
│   │   └── README.md             # Training documentation
│   ├── models/                   # Trained model files
│   │   ├── effnetb7_perfect_model.h5
│   │   └── ResUNet-bestSegModel-weights.h5
│   ├── train_model.py            # Model training utility
│   ├── test_model.py             # Model testing utility
│   └── plot_training.py          # Training visualization
└── venv/                         # Virtual environment
```

## 🎯 API Endpoints

### Authentication
- `POST /api/auth/register/` - User registration
- `POST /api/auth/login/` - User login

### Tumor Analysis
- `POST /api/analyze/` - Upload MRI scan for analysis

### User Management
- `GET /api/users/profile/` - Get user profile
- `PUT /api/users/profile/` - Update user profile

## 🔬 AI Models

### Classification Model (EfficientNetB7)
- **Architecture**: EfficientNetB7 with custom top layers
- **Input**: 224x224x3 RGB images
- **Output**: 4-class classification (glioma, meningioma, notumor, pituitary)
- **Performance**: High accuracy with optimized inference

### Segmentation Model (ResUNet)
- **Architecture**: Residual U-Net with skip connections
- **Input**: 256x256x3 RGB images
- **Output**: Binary segmentation mask
- **Features**: Precise tumor boundary detection

## 📊 Analysis Results

The API returns comprehensive analysis including:

- **Tumor Classification**: Type and confidence score
- **Segmentation Mask**: Precise tumor boundaries
- **Tumor Measurements**: Area, perimeter, diameter
- **Size Classification**: Small, medium, large
- **Visualization**: Overlay of segmentation on original image

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the `tumor_detection` directory:

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Model Paths
Update model paths in `api/views.py` if needed:
```python
CLASSIFICATION_MODEL_PATH = 'path/to/classification_model.h5'
SEGMENTATION_MODEL_PATH = 'path/to/segmentation_model.h5'
```

## 🚀 Deployment

### Production Setup

1. **Install production dependencies**
   ```bash
   pip install gunicorn psycopg2-binary
   ```

2. **Configure environment**
   ```bash
   export DEBUG=False
   export SECRET_KEY=your-production-secret-key
   ```

3. **Collect static files**
   ```bash
   python manage.py collectstatic
   ```

4. **Run with Gunicorn**
   ```bash
   gunicorn tumor_detection.wsgi:application
   ```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["gunicorn", "tumor_detection.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## 🧪 Testing

### Run Tests
```bash
python manage.py test
```

### Test Model Performance
```bash
python test_model.py
```

### API Testing
Use the built-in DRF browsable API or tools like Postman:
- URL: `http://localhost:8000/api/`
- Authentication: Token-based

## 📈 Model Training

### Training Pipeline
```bash
cd training
python train.py
```

### Training Features
- Data augmentation
- Class weight balancing
- Mixed precision training
- Early stopping
- Model checkpointing
- TensorBoard logging

See [training/README.md](tumor_detection/training/README.md) for detailed training documentation.

## 🔒 Security

- **Authentication**: Token-based authentication
- **CORS**: Configured for frontend integration
- **File Upload**: Secure file handling with validation
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: Built-in Django security features

## 📝 API Documentation

### Request Format
```json
{
  "image": "base64_encoded_image_or_file_upload"
}
```

### Response Format
```json
{
  "classification": {
    "tumor_type": "glioma",
    "confidence": 0.95
  },
  "segmentation": {
    "mask": "base64_encoded_mask",
    "area_mm2": 125.5,
    "perimeter_mm": 45.2,
    "diameter_mm": 12.8,
    "size_class": "medium"
  },
  "visualization": "base64_encoded_overlay_image"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support, email support@braintumordetection.com or create an issue in the repository.

## 🔗 Links

- [Frontend Repository](../Frontend/README.md)
- [API Documentation](http://localhost:8000/api/)
- [Django Admin](http://localhost:8000/admin/)

## 📊 Performance

- **Inference Time**: < 2 seconds per image
- **Model Accuracy**: > 95% classification accuracy
- **Memory Usage**: Optimized for production deployment
- **Scalability**: Ready for horizontal scaling

---

**Built with ❤️ for better healthcare through AI**

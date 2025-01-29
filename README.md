# Lung Opacity & COVID-19 Detection Using Deep Learning

## Overview
This project implements a three-layer Deep Neural Network (DNN) using TensorFlow to classify lung images into four categories:
- **COVID-19 Positive**
- **Viral Pneumonia**
- **Lung Opacity (Non-COVID lung infection)**
- **Normal Lungs**

The dataset used for training is the **"COVID-19 Radiography Dataset"**, with a specific focus on the "Lung Opacity (Non-COVID lung infection)" category, which includes various lung infections other than COVID-19. This classification is crucial for improving diagnostic accuracy and developing effective medical tools.

## Dataset
- The dataset is stored in the folder: **`covid19_radiography_dataset`**.
- The Jupyter Notebook file used for training the model: **`covid_lung_v1.ipynb`**.

## Model Architecture
- **Deep Learning Framework:** TensorFlow
- **Architecture:** Three-layer Deep Neural Network (DNN)
- **Cost Function:** Categorical cross-entropy cost function.
- **Training Details:**
  - **Epochs:** 800
  - **Final Cost:** 0.169048
  - **Training Accuracy:** 93.99%
  - **Testing Accuracy:** 77.86%
- **Input:** Preprocessed lung X-ray images
- **Output:** Predicted class (COVID, Pneumonia, Lung Opacity, or Normal)

The trained parameters were saved and used for deployment.

## Deployment
This project is deployed on **AWS EC2**, making it accessible via a public IP address.

### Steps Taken:
1. **Launched an EC2 instance** with Ubuntu.
2. **Configured security groups** to allow inbound traffic on ports **22, 80, and 5002**.
3. **Installed required dependencies** inside a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Deployed the Flask app** using Gunicorn as the WSGI server.
5. **Application is accessible** through the public IP address:
   ```
   http://16.171.1.219:5002
   ```

## Usage
1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/your-username/lung-opacity-detection.git
   cd lung-opacity-detection
   ```
2. Start the Flask app (if running locally):
   ```bash
   python app.py
   ```
3. Upload a lung X-ray image to get a prediction.


## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the **MIT License**.

---
### Contact
For questions or collaboration, reach out at **[your-email@example.com](mailto:your-email@example.com)**.

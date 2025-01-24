## Deployment

This project is deployed on **AWS EC2**. The app is hosted on a virtual machine with Ubuntu and is accessible via a public IP address: 16.171.1.219.

### Steps Taken:
1. Launched an EC2 instance with Ubuntu.
2. Configured security groups to open ports 22, 80, and 5002.
3. Installed required dependencies via a virtual environment (`python3 -m venv venv`).
4. Deployed the Flask app using Gunicorn as the WSGI server.
5. App is accessible through the public IP address and port `5002`.

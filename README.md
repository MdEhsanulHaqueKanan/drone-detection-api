---
title: Drone Detection API
sdk: docker
---

# ✈️ Drone Detection API (Backend)

This repository contains the containerized Flask backend for the **Drone Detection System**, a full-stack, decoupled computer vision application.

### Project Overview

This API serves a fine-tuned PyTorch (Faster R-CNN) model capable of detecting drones in user-uploaded images. It exposes a `/predict` endpoint that accepts an image and returns a JSON object containing a list of detected drones, each with its bounding box coordinates, label, and confidence score.

This backend is designed as a standalone microservice, containerized with Docker, and is currently deployed on Hugging Face Spaces to leverage their powerful free-tier hardware for ML inference.

### 🔗 Project Links

| Link                               | URL                                                                                                         |
| :--------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| 🚀 **Live Demo**                   | **[drone-detection-frontend.vercel.app](https://drone-detection-frontend.vercel.app/)** |
| 🎨 **Frontend Repository (React)** | [github.com/MdEhsanulHaqueKanan/drone-detection-frontend](https://github.com/MdEhsanulHaqueKanan/drone-detection-frontend) |
| ⚙️ **Backend API Repository (This Repo)** | [github.com/MdEhsanulHaqueKanan/drone-detection-api](https://github.com/MdEhsanulHaqueKanan/drone-detection-api)       |
| 📦 **Original Monolithic Project** | [github.com/MdEhsanulHaqueKanan/drone-detection-deep-learning-flask-app](https://github.com/MdEhsanulHaqueKanan/drone-detection-deep-learning-flask-app) |

**Note on Live Demo:** This API is hosted on Hugging Face's free community tier. If the app has been inactive, it may "sleep" to save resources. The first prediction might take **30-90 seconds** as the server wakes up. Subsequent predictions will be much faster!

### Technology Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** PyTorch, Torchvision, Albumentations
*   **Containerization:** Docker
*   **Large File Storage:** Git LFS
*   **Deployment:** Hugging Face Spaces
# 🦅 Indian Birds Species Classifier

An AI-powered computer vision application for identifying **Indian bird species** from uploaded images.

The application uses a fine-tuned **ResNet50V2** deep learning model and returns the **Top 5 predicted species** together with confidence scores.

## 🚀 Live Demo

👉 [Open the Indian Birds Species Classifier on Hugging Face](https://huggingface.co/spaces/ferides/indian-birds-species-classifier)

## 🌿 Project Overview

This project demonstrates how deep learning and computer vision can be used for wildlife image classification.

Users can upload a bird image, and the trained model analyzes the image to identify the most likely bird species.

The classifier currently supports **25 Indian bird species**.

## 🐦 Features

- Upload bird images
- Classify Indian bird species
- Display Top 5 predictions
- Show confidence scores
- Display the most likely species
- Additional notes for selected species
- Interactive Gradio web interface
- Nature-inspired user interface

## 🧠 Model

The classifier is based on:

**ResNet50V2**

Image preprocessing:

```text
Input Image
    ↓
RGB Conversion
    ↓
Resize to 224 × 224
    ↓
ResNet50V2 Preprocessing
    ↓
Deep Learning Model
    ↓
Top 5 Predictions

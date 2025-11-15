# Bankruptcy Prevention Risk Analyzer

A bankruptcy prediction web app built with Streamlit. It predicts whether a company is at risk of bankruptcy based on financial and operational factors.

## What is this?

This app analyzes 6 key financial factors and uses machine learning to predict if a company might go bankrupt. It's like an early warning system for financial trouble. You input values for things like management quality, financial flexibility, and market competitiveness, and the app tells you how risky the company is.

## How it works

The app uses two types of models:
- An ensemble model that combines 7 different ML models (Logistic Regression, KNN, Random Forest, Decision Tree, SVM, XGBoost, and LightGBM)
- A single best model (KNN) for comparison

The models are trained on data from 250 companies. When you enter company metrics, the app automatically creates 16 features from your 6 inputs and feeds them into the model.

## What you can input

The app asks you to rate these factors on a scale from 0 to 1:
- Industrial Risk: How risky the industry is
- Management Risk: Quality of management
- Financial Flexibility: Can they adapt to financial challenges
- Credibility: How trustworthy is the company
- Competitiveness: How well they compete in the market
- Operating Risk: How efficient they are

## Installation

1. Make sure you have Python 3.8 or higher installed

2. Clone or download this project

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the app:
   ```
   streamlit run prediction.py
   ```

The app will open in your browser at http://localhost:8501

## Files in this project

- prediction.py - The main app
- Bankruptcy Prevention-1.ipynb - Jupyter notebook with model training code
- bankruptcy_with_features.csv - Dataset with engineered features
- Bankruptcy.xlsx - Original data
- models/ - Folder with trained model files
- requirements.txt - All the Python packages needed

## Understanding the results

The app shows you three things:
- A risk level: Low, Medium, or High
- Bankruptcy probability: How likely it is (as a percentage)
- Success probability: How likely they are to be fine

Based on your score, it also gives you recommendations. High risk companies get immediate action items, medium risk gets preventive measures, and low risk gets strategies to stay healthy.

## If things don't work

Model files missing: Make sure the models/ folder is in the same place as prediction.py and has all the .pkl files

Dependencies not installed: Run `pip install -r requirements.txt` again

File paths wrong: Make sure you're running the app from the project root directory

## Technologies used

- Streamlit (web app)
- Python (programming)
- scikit-learn, XGBoost, LightGBM and other Machine Learning models(machine learning)
- pandas, NumPy (data processing)

## Notes

- This is trained on 250 companies, so results are best for companies similar to those in the training data
- The app doesn't store any data you input

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import time

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="🔍 Bankruptcy Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ========================================
# INITIAL PAGE LOADING SCREEN (ONLY FIRST TIME)
# ========================================
# This loading screen will appear when:
# 1. User first visits the deployed app online
# 2. Browser session is refreshed/restarted
# 3. Streamlit reruns due to code changes (in development)
# It will NOT appear on subsequent interactions (input changes, button clicks)
# This provides optimal UX for deployed production apps

# Initialize session state to track if app has been loaded
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False

# Show loading screen only on first load (critical for deployment)
if not st.session_state.app_loaded:
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        st.markdown("""
            <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.7; transform: scale(0.95); }
            }
            
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes progressBar {
                0% { width: 0%; }
                100% { width: 100%; }
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            @keyframes gradientShift {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            
            .initial-loading-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                background-size: 400% 400%;
                animation: gradientShift 3s ease infinite;
                z-index: 9999;
            }
            
            .loader-wrapper {
                position: relative;
                margin-bottom: 40px;
            }
            
            .loader {
                width: 100px;
                height: 100px;
                border: 10px solid rgba(255, 255, 255, 0.2);
                border-top: 10px solid white;
                border-radius: 50%;
                animation: spin 1.2s linear infinite;
            }
            
            .loader-inner {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 40px;
                animation: float 2s ease-in-out infinite;
            }
            
            .loading-text {
                color: white;
                font-size: 32px;
                font-weight: 800;
                text-align: center;
                animation: pulse 2s ease-in-out infinite;
                margin-bottom: 15px;
                text-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            }
            
            .loading-subtext {
                color: rgba(255, 255, 255, 0.95);
                font-size: 18px;
                text-align: center;
                animation: slideUp 0.8s ease-out;
                max-width: 600px;
                padding: 0 20px;
                line-height: 1.6;
                margin-bottom: 30px;
            }
            
            .progress-container-initial {
                width: 400px;
                max-width: 90%;
                height: 6px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 20px;
            }
            
            .progress-bar-initial {
                height: 100%;
                background: linear-gradient(90deg, #ffffff 0%, #f0f0f0 100%);
                border-radius: 10px;
                animation: progressBar 2s ease-in-out;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
            }
            
            .loading-dots {
                display: flex;
                gap: 10px;
                margin-top: 25px;
            }
            
            .dot {
                width: 14px;
                height: 14px;
                background: white;
                border-radius: 50%;
                animation: pulse 1.5s ease-in-out infinite;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
            }
            
            .dot:nth-child(2) { animation-delay: 0.3s; }
            .dot:nth-child(3) { animation-delay: 0.6s; }
            
            .loading-status {
                color: rgba(255, 255, 255, 0.85);
                font-size: 14px;
                text-align: center;
                margin-top: 15px;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            </style>
            
            <div class="initial-loading-container">
                <div class="loader-wrapper">
                    <div class="loader"></div>
                    <div class="loader-inner">🔍</div>
                </div>
                <div class="loading-text">Bankruptcy Risk Analyzer</div>
                <div class="loading-subtext">
                    Loading AI-powered financial analytics platform...<br>
                    Initializing machine learning models and workspace
                </div>
                <div class="progress-container-initial">
                    <div class="progress-bar-initial"></div>
                </div>
                <div class="loading-status">⚡ Loading prediction models...</div>
                <div class="loading-dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
else:
    # Create empty placeholder for when app is already loaded
    loading_placeholder = None

# Advanced CSS with animations and responsive design
st.markdown("""
    <style>
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7);
        }
        50% {
            box-shadow: 0 0 0 20px rgba(231, 76, 60, 0);
        }
    }
    
    @keyframes pulseGreen {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.7);
        }
        50% {
            box-shadow: 0 0 0 20px rgba(39, 174, 96, 0);
        }
    }
    
    @keyframes pulseOrange {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(243, 156, 18, 0.7);
        }
        50% {
            box-shadow: 0 0 0 20px rgba(243, 156, 18, 0);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.8),
                         0 0 20px rgba(255, 255, 255, 0.6);
        }
        50% {
            text-shadow: 0 0 20px rgba(255, 255, 255, 1),
                         0 0 30px rgba(255, 255, 255, 0.8),
                         0 0 40px rgba(255, 255, 255, 0.6);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    @keyframes scaleUp {
        from {
            transform: scale(0.8);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-100%);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 0;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header Styles */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: slideDown 1s ease-out, gradientMove 4s ease infinite;
        padding: 50px 20px;
        border-radius: 0 0 30px 30px;
        margin-bottom: 40px;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    @keyframes gradientMove {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent,
            rgba(255, 255, 255, 0.1),
            transparent
        );
        animation: shimmer 3s infinite;
    }
    
    .header-title {
        color: white;
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        margin: 0;
        animation: glow 2s ease-in-out infinite, float 3s ease-in-out infinite;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5),
                     0 0 30px rgba(255, 255, 255, 0.3);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        text-align: center;
        font-size: 20px;
        margin-top: 15px;
        animation: fadeInUp 1s ease-out 0.3s both;
        position: relative;
        z-index: 1;
        letter-spacing: 1px;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 20px;
        border-radius: 15px;
        border: none;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: slideInUp 0.8s ease-out 0.3s both, gradientMove 3s ease infinite;
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.03);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.5);
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        margin: 15px 0;
        animation: bounceIn 0.8s ease-out;
        border-top: 5px solid #667eea;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(102, 126, 234, 0.1),
            transparent
        );
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.3);
        border-top-color: #764ba2;
    }
    
    /* Risk Badge Styles */
    .risk-badge-high {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 50%, #e74c3c 100%);
        background-size: 200% 200%;
        color: white;
        padding: 50px 30px;
        border-radius: 25px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(231, 76, 60, 0.4);
        animation: scaleUp 0.8s ease-out, pulse 2s ease-in-out infinite, gradientMove 3s ease infinite;
        border: 3px solid rgba(255, 255, 255, 0.3);
        position: relative;
    }
    
    .risk-badge-high::after {
        content: '⚠️';
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 30px;
        animation: rotate 3s linear infinite;
    }
    
    .risk-badge-medium {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 50%, #f39c12 100%);
        background-size: 200% 200%;
        color: white;
        padding: 50px 30px;
        border-radius: 25px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(243, 156, 18, 0.4);
        animation: scaleUp 0.8s ease-out, pulseOrange 2s ease-in-out infinite, gradientMove 3s ease infinite;
        border: 3px solid rgba(255, 255, 255, 0.3);
        position: relative;
    }
    
    .risk-badge-medium::after {
        content: '⚡';
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 30px;
        animation: float 2s ease-in-out infinite;
    }
    
    .risk-badge-low {
        background: linear-gradient(135deg, #27ae60 0%, #229954 50%, #1e8449 100%);
        background-size: 200% 200%;
        color: white;
        padding: 50px 30px;
        border-radius: 25px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(39, 174, 96, 0.4);
        animation: scaleUp 0.8s ease-out, pulseGreen 2s ease-in-out infinite, gradientMove 3s ease infinite;
        border: 3px solid rgba(255, 255, 255, 0.3);
        position: relative;
    }
    
    .risk-badge-low::after {
        content: '✨';
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 30px;
        animation: rotate 4s linear infinite;
    }
    
    .risk-badge-title {
        font-size: 48px;
        font-weight: 900;
        margin: 10px 0;
        letter-spacing: 2px;
        animation: float 3s ease-in-out infinite;
        text-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    .risk-badge-prob {
        font-size: 36px;
        margin: 20px 0;
        font-weight: 800;
        text-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .risk-badge-prob .glow-number {
        color: #ffffff;
        font-size: 42px;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.8),
                     0 0 30px rgba(255, 255, 255, 0.6),
                     0 4px 10px rgba(0, 0, 0, 0.3);
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Input Section */
    .input-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 45px;
        border-radius: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.12);
        animation: fadeInUp 0.8s ease-out;
        border-top: 5px solid #667eea;
        position: relative;
        overflow: hidden;
    }
    
    .input-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.05) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    .input-section h3 {
        color: #2c3e50;
        margin-bottom: 35px;
        font-size: 32px;
        font-weight: 800;
        position: relative;
        z-index: 1;
        animation: slideInLeft 0.8s ease-out;
    }
    
    /* Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 40px;
        border-radius: 25px;
        margin: 30px 0;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
        border-left: 10px solid #667eea;
        animation: slideInLeft 0.8s ease-out;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 0;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        transition: height 0.5s ease;
    }
    
    .recommendation-box:hover::before {
        height: 100%;
    }
    
    .recommendation-box:hover {
        transform: translateX(15px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.18);
    }
    
    .recommendation-box h4 {
        margin-bottom: 25px;
        font-size: 26px;
        font-weight: 800;
    }
    
    .recommendation-box p {
        color: #2c3e50;
        line-height: 2;
        font-size: 17px;
        margin: 12px 0;
    }
    
    /* Results Container */
    .results-container {
        margin-top: 50px;
        animation: fadeInUp 1s ease-out 0.4s both;
    }
    
    /* Progress Bar Animation */
    @keyframes progressBar {
        from { width: 0%; }
        to { width: 100%; }
    }
    
    .progress-container {
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 20px 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        animation: progressBar 2s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 36px;
        }
        
        .header-subtitle {
            font-size: 16px;
        }
        
        .header-container {
            padding: 30px 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            padding: 20px;
            margin: 10px 0;
        }
        
        .risk-badge-title {
            font-size: 32px;
        }
        
        .risk-badge-prob {
            font-size: 24px;
        }
        
        .input-section {
            padding: 20px;
        }
        
        .recommendation-box {
            padding: 20px;
        }
        
        .recommendation-box h4 {
            font-size: 18px;
        }
        
        .recommendation-box p {
            font-size: 14px;
        }
        
        .stButton>button {
            font-size: 16px;
            padding: 15px;
        }
    }
    
    @media (max-width: 480px) {
        .header-title {
            font-size: 28px;
        }
        
        .header-container {
            padding: 25px 10px;
            border-radius: 0;
        }
        
        .risk-badge-title {
            font-size: 24px;
        }
        
        .risk-badge-prob {
            font-size: 20px;
        }
    }
    
    /* Text Animations */
    .animate-title {
        animation: fadeInDown 0.8s ease-out;
    }
    
    .animate-subtitle {
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    /* Sidebar Styles */
    .sidebar {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Section Dividers */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 40px 0;
        border-radius: 2px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #ecf0f1;
        padding: 50px 20px;
        margin-top: 80px;
        border-top: 3px solid rgba(102, 126, 234, 0.3);
        animation: fadeInUp 1s ease-out 0.8s both;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 20px;
    }
    
    .footer p {
        margin: 12px 0;
        font-size: 15px;
    }
    
    /* Glow Effect for Numbers */
    .glow-number {
        color: #ffffff;
        font-weight: 900;
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Particle Effect */
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 50%;
        animation: particleFloat 3s ease-in-out infinite;
    }
    
    @keyframes particleFloat {
        0%, 100% {
            transform: translateY(0) translateX(0);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(-100px) translateX(50px);
            opacity: 0;
        }
    }
    
    /* Glow Effect for Numbers */
    .glow-number {
        color: #ffffff;
        font-weight: 900;
        animation: glow 2s ease-in-out infinite;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Feature engineering function
def create_features(df):
    """Create engineered features from basic features"""
    df_featured = df.copy()
    
    risk_cols = ['industrial_risk', 'management_risk', 'operating_risk']
    financial_cols = ['financial_flexibility', 'credibility', 'competitiveness']
    
    df_featured['financial_health_score'] = df_featured[financial_cols].mean(axis=1)
    df_featured['management_impact_score'] = df_featured['management_risk'] / (df_featured['financial_flexibility'] + df_featured['credibility'] + 1)
    df_featured['risk_stability_ratio'] = (df_featured['financial_flexibility'] + df_featured['credibility']) / (df_featured['management_risk'] + 1)
    
    df_featured['risk_volatility'] = df_featured[risk_cols].std(axis=1)
    weights = {'financial_flexibility': 0.4, 'credibility': 0.3, 'competitiveness': 0.3}
    df_featured['financial_stability'] = sum(df_featured[col] * w for col, w in weights.items())
    df_featured['risk_financial_ratio'] = df_featured[risk_cols].mean(axis=1) / (df_featured[financial_cols].mean(axis=1) + 1)
    
    df_featured['management_financial_risk'] = df_featured['management_risk'] / (df_featured['financial_flexibility'] + 0.1)
    df_featured['operational_sustainability'] = ((df_featured['financial_flexibility'] + df_featured['competitiveness']) / 2) * (1 - df_featured['operating_risk'])
    risk_threshold = 0.7
    df_featured['compound_risk'] = (df_featured[risk_cols] > risk_threshold).sum(axis=1) / len(risk_cols)
    
    df_featured['financial_x_management'] = df_featured['financial_health_score'] * df_featured['management_risk']
    df_featured['risk_x_operational'] = df_featured['risk_volatility'] * df_featured['operating_risk']
    
    return df_featured

# Load models
@st.cache_resource
def load_models_and_metadata():
    models_dir = "models"
    ensemble_model = None
    best_model = None
    scaler = None
    ensemble_metadata = None
    best_model_metadata = None
    
    try:
        # 1. Load ENSEMBLE model (optimized - removed verbose prints)
        ensemble_file = os.path.join(models_dir, "ensemble_model.pkl")
        if os.path.exists(ensemble_file):
            try:
                ensemble_model = joblib.load(ensemble_file)
            except Exception:
                with open(ensemble_file, 'rb') as file:
                    ensemble_model = pickle.load(file)
        else:
            # Fallback to root directory
            if os.path.exists('bankruptcy_ensemble_model.pkl'):
                try:
                    ensemble_model = joblib.load('bankruptcy_ensemble_model.pkl')
                except Exception:
                    with open('bankruptcy_ensemble_model.pkl', 'rb') as file:
                        ensemble_model = pickle.load(file)
        
        # 2. Load BEST SINGLE model (KNN)
        best_model_files = [f for f in os.listdir(models_dir) if f.startswith("best_model_")]
        if best_model_files:
            best_model_file = os.path.join(models_dir, best_model_files[0])
            try:
                best_model = joblib.load(best_model_file)
            except Exception:
                with open(best_model_file, 'rb') as file:
                    best_model = pickle.load(file)
        
        # 3. Load ensemble metadata
        ensemble_metadata_file = os.path.join(models_dir, "ensemble_metadata.pkl")
        if os.path.exists(ensemble_metadata_file):
            with open(ensemble_metadata_file, 'rb') as file:
                ensemble_metadata = pickle.load(file)
        
        # 4. Load best model metadata
        best_model_metadata_file = os.path.join(models_dir, "model_metadata.pkl")
        if os.path.exists(best_model_metadata_file):
            with open(best_model_metadata_file, 'rb') as file:
                best_model_metadata = pickle.load(file)
        
        # 5. Load scaler - try both joblib and pickle
        scaler_file = os.path.join(models_dir, "feature_scaler.pkl")
        if os.path.exists(scaler_file):
            try:
                scaler = joblib.load(scaler_file)
            except Exception:
                with open(scaler_file, 'rb') as file:
                    scaler = pickle.load(file)
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"❌ Error loading models: {str(e)}")
        with st.expander("Show detailed error"):
            st.code(error_details)
        
    return ensemble_model, best_model, scaler, ensemble_metadata, best_model_metadata

# ========================================
# MODEL LOADING (WITH SESSION STATE CACHING)
# ========================================
# Models are loaded once per user session and cached in st.session_state
# This is critical for deployment as it:
# 1. Prevents reloading models on every interaction (faster response)
# 2. Reduces server load when deployed online
# 3. Improves user experience with instant updates after initial load

# Load all models and store in session state (only once per session)
if 'ensemble_model' not in st.session_state:
    # First time loading - models will load while loading screen is displayed
    ensemble_model, best_model, scaler, ensemble_metadata, best_model_metadata = load_models_and_metadata()
    st.session_state.ensemble_model = ensemble_model
    st.session_state.best_model = best_model
    st.session_state.scaler = scaler
    st.session_state.ensemble_metadata = ensemble_metadata
    st.session_state.best_model_metadata = best_model_metadata
    
    # Add delay and clear loading screen after models are loaded
    if loading_placeholder:
        time.sleep(0.7)  # Reduced from 1.2s - faster UI load
        loading_placeholder.empty()
    
    # Mark app as fully loaded
    st.session_state.app_loaded = True
else:
    # Retrieve from session state on subsequent reruns
    ensemble_model = st.session_state.ensemble_model
    best_model = st.session_state.best_model
    scaler = st.session_state.scaler
    ensemble_metadata = st.session_state.ensemble_metadata
    best_model_metadata = st.session_state.best_model_metadata

# Verify at least one model is loaded
if ensemble_model is None and best_model is None:
    st.error("⚠️ **Critical Error**: No models could be loaded!")
    st.info("Please ensure you have run the notebook cells to generate the models.")
    st.stop()

# Verify at least one model is loaded
if ensemble_model is None and best_model is None:
    st.error("⚠️ **Critical Error**: No models could be loaded!")
    st.info("Please ensure you have run the notebook cells to generate the models.")
    st.stop()

# Header Section
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🔍 BANKRUPTCY RISK ANALYZER</h1>
        <p class="header-subtitle">AI-Powered Financial Risk Assessment & Predictive Analytics Platform</p>
        <div class="progress-container">
            <div class="progress-bar"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with model selection and info
with st.sidebar:
    st.markdown("### 🤖 Model Selection")
    
    # Create model selection options
    model_options = []
    if ensemble_model:
        model_options.append("Ensemble (7 Models)")
    if best_model:
        best_model_name = best_model_metadata.get('model_name', 'Best Single Model') if best_model_metadata else 'Best Single Model'
        model_options.append(f"Best Single Model ({best_model_name})")
    
    # Default to ensemble if available, otherwise best model
    default_option = model_options[0] if model_options else None
    
    selected_model_option = st.selectbox(
        "Choose Prediction Model:",
        options=model_options,
        index=0,
        help="Select which model to use for bankruptcy prediction"
    )
    
    # Determine which model to use
    use_ensemble = "Ensemble" in selected_model_option
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    # Display information based on selected model
    if use_ensemble and ensemble_metadata:
        metadata = ensemble_metadata
        performance = metadata['performance']
        model_name = metadata.get('model_name', 'Ensemble Model')
        
        # Check if this is an ensemble model
        if 'ensemble_models' in metadata:
            num_models = metadata.get('num_models', len(metadata['ensemble_models']))
            voting_type = metadata.get('voting_type', 'soft')
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); padding: 20px; border-radius: 15px; margin: 15px 0; border: 1px solid rgba(255,255,255,0.2);'>
                <p style='margin: 0; color: rgba(255,255,255,0.7); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;'>Active Model</p>
                <p style='margin: 10px 0 0 0; font-size: 20px; font-weight: bold; color: white;'>{model_name}</p>
                <p style='margin: 10px 0 0 0; font-size: 14px; color: rgba(255,255,255,0.8);'>🔗 {num_models} Models Combined</p>
                <p style='margin: 5px 0 0 0; font-size: 12px; color: rgba(255,255,255,0.6);'>Voting: {voting_type.capitalize()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display constituent models
            st.markdown("#### 🤖 Ensemble Components")
            for i, model in enumerate(metadata['ensemble_models'], 1):
                st.markdown(f"**{i}.** {model.replace('_', ' ').title()}")
    
    elif not use_ensemble and best_model_metadata:
        metadata = best_model_metadata
        performance = metadata['performance']
        model_name = metadata.get('model_name', 'Best Single Model')
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); padding: 20px; border-radius: 15px; margin: 15px 0; border: 1px solid rgba(255,255,255,0.2);'>
            <p style='margin: 0; color: rgba(255,255,255,0.7); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;'>Active Model</p>
            <p style='margin: 10px 0 0 0; font-size: 20px; font-weight: bold; color: white;'>{model_name}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback if no metadata available
        st.warning("⚠️ Model metadata not available")
        metadata = None
    
    # Display performance metrics if metadata is available
    if metadata and 'performance' in metadata:
        performance = metadata['performance']
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Accuracy", f"{performance['accuracy']:.1%}", delta=None)
            st.metric("Precision", f"{performance['precision']:.1%}", delta=None)
        with col2:
            st.metric("F1-Score", f"{performance['f1_score']:.1%}", delta=None)
            st.metric("Recall", f"{performance['recall']:.1%}", delta=None)

# Input Section
st.markdown("""
    <div class="input-section">
        <h3 class="animate-title">📝 Enter Company Risk Factors</h3>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    industrial_risk = st.selectbox('🏭 Industrial Risk', options=['0.0 - Low', '0.5 - Medium', '1.0 - High'], index=1)
    industrial_risk = float(industrial_risk.split(' -')[0])
    
    management_risk = st.selectbox('👔 Management Risk', options=['0.0 - Low', '0.5 - Medium', '1.0 - High'], index=1)
    management_risk = float(management_risk.split(' -')[0])

with col2:
    financial_flexibility = st.selectbox('💰 Financial Flexibility', options=['0.0 - Low', '0.5 - Medium', '1.0 - High'], index=1)
    financial_flexibility = float(financial_flexibility.split(' -')[0])
    
    credibility = st.selectbox('🎯 Credibility', options=['0.0 - Low', '0.5 - Medium', '1.0 - High'], index=1)
    credibility = float(credibility.split(' -')[0])

with col3:
    competitiveness = st.selectbox('🚀 Competitiveness', options=['0.0 - Low', '0.5 - Medium', '1.0 - High'], index=1)
    competitiveness = float(competitiveness.split(' -')[0])
    
    operating_risk = st.selectbox('⚙️ Operating Risk', options=['0.0 - Low', '0.5 - Medium', '1.0 - High'], index=1)
    operating_risk = float(operating_risk.split(' -')[0])

st.markdown("<br>", unsafe_allow_html=True)

# Predict button
if st.button('🔮 ANALYZE BANKRUPTCY RISK', type='primary'):
    # Create a placeholder for prediction loading
    prediction_loading = st.empty()
    
    # Show loading animation in the prediction area
    with prediction_loading.container():
        st.markdown("""
            <style>
            @keyframes spinPrediction {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes pulsePrediction {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(0.98); }
            }
            
            .prediction-loading-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding: 80px 20px;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 25px;
                margin: 40px 0;
                animation: pulsePrediction 2s ease-in-out infinite;
            }
            
            .prediction-spinner {
                width: 60px;
                height: 60px;
                border: 6px solid rgba(102, 126, 234, 0.2);
                border-top: 6px solid #667eea;
                border-radius: 50%;
                animation: spinPrediction 1s linear infinite;
                margin-bottom: 25px;
            }
            
            .prediction-loading-text {
                color: #667eea;
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 10px;
                text-align: center;
            }
            
            .prediction-loading-subtext {
                color: #7f8c8d;
                font-size: 16px;
                text-align: center;
                max-width: 400px;
            }
            </style>
            
            <div class="prediction-loading-container">
                <div class="prediction-spinner"></div>
                <div class="prediction-loading-text">🔮 Analyzing Bankruptcy Risk...</div>
                <div class="prediction-loading-subtext">
                    Processing financial data and generating predictions
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Select the active model based on user choice
    if use_ensemble:
        active_model = ensemble_model
        active_metadata = ensemble_metadata
        model_display_name = "Ensemble (7 Models)"
    else:
        active_model = best_model
        active_metadata = best_model_metadata
        model_display_name = best_model_metadata.get('model_name', 'Best Single Model') if best_model_metadata else 'Best Single Model'
    
    # Verify model is loaded
    if active_model is None:
        prediction_loading.empty()
        st.error(f"❌ {model_display_name} not loaded! Please ensure the model files exist in the 'models' directory.")
        st.stop()
    
    if scaler is None:
        st.warning("⚠️ Scaler not found. Using default scaling.")
        scaler = MinMaxScaler()
    
    # Prepare data
    basic_input = pd.DataFrame({
        'industrial_risk': [industrial_risk],
        'management_risk': [management_risk],
        'financial_flexibility': [financial_flexibility],
        'credibility': [credibility],
        'competitiveness': [competitiveness],
        'operating_risk': [operating_risk]
    })
    
    input_featured = create_features(basic_input)
    
    if active_metadata and 'features' in active_metadata:
        input_features = active_metadata['features']
        for feat in input_features:
            if feat not in input_featured.columns:
                input_featured[feat] = 0
        input_data = input_featured[input_features]
    else:
        input_data = input_featured.select_dtypes(include=np.number)
    
    if scaler:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data.values
    
    # Make predictions
    prediction = active_model.predict(input_scaled)[0]
    probability = active_model.predict_proba(input_scaled)[0]
    
    bankruptcy_prob = probability[0]
    non_bankruptcy_prob = probability[1]
    
    # Minimal delay for smooth transition - optimized for speed
    time.sleep(0.2)  # Reduced from 0.5s
    
    # Clear prediction loading animation
    prediction_loading.empty()
    
    # Results section
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("""
        <h2 style='text-align: center; color: #ecf0f1; margin-bottom: 50px; font-size: 42px; font-weight: 900; animation: fadeInDown 0.8s ease-out;'>
            📊 COMPREHENSIVE RISK ANALYSIS
        </h2>
    """, unsafe_allow_html=True)
    
    # Determine risk level
    if bankruptcy_prob > 0.7:
        risk_level = "HIGH RISK"
        color = "#e74c3c"
        icon = "🔴"
        badge_class = "risk-badge-high"
    elif bankruptcy_prob > 0.4:
        risk_level = "MEDIUM RISK"
        color = "#f39c12"
        icon = "🟠"
        badge_class = "risk-badge-medium"
    else:
        risk_level = "LOW RISK"
        color = "#27ae60"
        icon = "🟢"
        badge_class = "risk-badge-low"
    
    # Main risk badge with animation
    st.markdown(f"""
    <div class="{badge_class}">
        <div style='font-size: 80px; margin-bottom: 20px; animation: bounceIn 1s ease-out;'>{icon}</div>
        <div class="risk-badge-title">{risk_level}</div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {bankruptcy_prob*100}%;"></div>
        </div>
        <div class="risk-badge-prob">
            <span class="glow-number">{bankruptcy_prob:.1%}</span> Bankruptcy Probability
        </div>
        <div style='font-size: 22px; font-weight: 600; margin-top: 15px; color: rgba(255,255,255,0.95);'>
            <span class="glow-number">{non_bankruptcy_prob:.1%}</span> Success Probability
        </div>
        <div style='margin-top: 25px; font-size: 16px; color: rgba(255,255,255,0.85); font-style: italic;'>
            Analysis completed with {model_display_name} • Confidence Score: {max(bankruptcy_prob, non_bankruptcy_prob):.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics in cards
    st.markdown("<h3 style='text-align: center; color: #ecf0f1; margin: 50px 0 40px 0; font-size: 36px; font-weight: 800; animation: fadeInUp 0.8s ease-out;'>📊 Key Performance Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style='color: #7f8c8d; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px;'>Risk Classification</h4>
            <h2 style='color: {color}; margin: 20px 0; font-size: 36px; font-weight: 900;'>{risk_level}</h2>
            <div style='width: 100%; height: 4px; background: {color}; border-radius: 2px; margin-top: 15px;'></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style='color: #7f8c8d; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px;'>Bankruptcy Risk</h4>
            <h2 style='color: {color}; margin: 20px 0; font-size: 36px; font-weight: 900;'>{bankruptcy_prob:.1%}</h2>
            <div style='width: {bankruptcy_prob*100}%; height: 4px; background: {color}; border-radius: 2px; margin-top: 15px; transition: width 1s ease;'></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        prediction_text = "Bankruptcy" if prediction == 0 else "Safe"
        prediction_icon = "⚠️" if prediction == 0 else "✅"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style='color: #7f8c8d; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px;'>Model Prediction</h4>
            <h2 style='color: {color}; margin: 20px 0; font-size: 36px; font-weight: 900;'>{prediction_icon} {prediction_text}</h2>
            <div style='width: 100%; height: 4px; background: {color}; border-radius: 2px; margin-top: 15px;'></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #ecf0f1; margin: 50px 0 40px 0; font-size: 36px; font-weight: 800; animation: fadeInUp 0.8s ease-out;'>💡 Strategic Recommendations</h2>", unsafe_allow_html=True)
    
    if bankruptcy_prob > 0.7:
        st.markdown(f"""
        <div class="recommendation-box" style="border-left-color: #e74c3c;">
            <h4 style='color: #e74c3c;'>⚠️ Immediate Actions Required</h4>
            <p><strong style='color: #2c3e50;'>Critical Priority:</strong></p>
            <p style='color: #2c3e50; line-height: 2;'>
                🔴 <strong>Strengthen Financial Flexibility</strong> - Secure emergency funding, negotiate with creditors<br>
                🔴 <strong>Improve Credibility</strong> - Maintain transparent communication with stakeholders<br>
                🔴 <strong>Reduce Operating Risk</strong> - Cut non-essential costs, optimize operations<br>
                🔴 <strong>Review Management Strategy</strong> - Consider bringing in turnaround specialists<br>
                🔴 <strong>Boost Competitiveness</strong> - Focus on core profitable products/services<br>
                🔴 <strong>Industry Analysis</strong> - Explore partnerships or market repositioning
            </p>
            <p style='font-size: 14px; color: #7f8c8d; margin-top: 20px;'><em>✓ Recommendation: Seek professional financial advisory services immediately.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
    elif bankruptcy_prob > 0.4:
        st.markdown(f"""
        <div class="recommendation-box" style="border-left-color: #f39c12;">
            <h4 style='color: #f39c12;'>⚡ Preventive Measures Recommended</h4>
            <p><strong style='color: #2c3e50;'>Action Plan:</strong></p>
            <p style='color: #2c3e50; line-height: 2;'>
                🟠 <strong>Enhance Financial Flexibility</strong> - Build cash reserves, diversify funding sources<br>
                🟠 <strong>Strengthen Credibility</strong> - Improve credit ratings, maintain good relationships<br>
                🟠 <strong>Manage Operating Risk</strong> - Implement risk management frameworks<br>
                🟠 <strong>Optimize Management</strong> - Review and refine business strategies<br>
                🟠 <strong>Increase Competitiveness</strong> - Invest in innovation and market differentiation<br>
                🟠 <strong>Monitor Industry Trends</strong> - Stay ahead of sector challenges
            </p>
            <p style='font-size: 14px; color: #7f8c8d; margin-top: 20px;'><em>✓ Recommendation: Regular quarterly financial reviews and proactive risk management.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
        <div class="recommendation-box" style="border-left-color: #27ae60;">
            <h4 style='color: #27ae60;'>✅ Strategies to Maintain Financial Health</h4>
            <p><strong style='color: #2c3e50;'>Best Practices:</strong></p>
            <p style='color: #2c3e50; line-height: 2;'>
                🟢 <strong>Maintain Financial Flexibility</strong> - Continue building reserves and maintaining liquidity<br>
                🟢 <strong>Sustain Credibility</strong> - Keep strong credit profile and stakeholder trust<br>
                🟢 <strong>Monitor Operating Risk</strong> - Regular operational audits and efficiency improvements<br>
                🟢 <strong>Effective Management</strong> - Continue strategic planning and execution excellence<br>
                🟢 <strong>Stay Competitive</strong> - Invest in R&D, customer satisfaction, and market leadership<br>
                🟢 <strong>Industry Leadership</strong> - Capitalize on market opportunities
            </p>
            <p style='font-size: 14px; color: #7f8c8d; margin-top: 20px;'><em>✓ Recommendation: Continue current practices while exploring growth opportunities.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p style='font-size: 20px; font-weight: 700; color: #ecf0f1; margin-bottom: 15px;'>🎯 Bankruptcy Prevention & Risk Management</p>
        <p style='font-size: 16px; color: #bdc3c7;'>Powered by Advanced AI & Machine Learning | © 2025</p>
        <p style='font-size: 13px; color: #95a5a6; margin-top: 10px;'>Built with Streamlit • Python • Scikit-learn • Advanced Analytics</p>
        <div style='margin-top: 20px; font-size: 12px; color: #7f8c8d;'>
            <span style='margin: 0 10px;'>⚡ Real-time Analysis</span>
            <span style='margin: 0 10px;'>🔒 Secure Processing</span>
            <span style='margin: 0 10px;'>📊 Data-Driven Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

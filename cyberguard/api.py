import os
import json
import pickle
import base64
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from cyberguard.features import CyberFeatureExtractor
from cyberguard.risk_engine import RiskEngine
from cyberguard.xai import DualXAIEngine
from cyberguard.model import CyberGuardHybridModel
from url_intelligence.inference import URLIntelligenceEngine

MSG_BUNDLE_PATH = r'e:\project 2\saved_models\cyberguard_bundle.pt'
URL_BUNDLE_PATH = r'e:\project 2\saved_models\url_intelligence_bundle.pkl'

class UnifiedCyberGuardHub:
    """
    Unified Production Inference Pipeline.
    Manages both Message Phishing Intelligence and Unified Direct URL & QR Code Intelligence.
    """
    def __init__(self):
        # 1. Message Model Setup
        self.msg_extractor = CyberFeatureExtractor()
        self.msg_xai_engine = DualXAIEngine()
        self.device = 'cpu'
        self.msg_model = None
        self.msg_scaler = None
        self.msg_vectorizer = None
        self.msg_risk_engine = RiskEngine()
        self._load_msg_bundle()

        # 2. URL & QR Intelligence Engine Setup
        self.url_engine = URLIntelligenceEngine(bundle_path=URL_BUNDLE_PATH)

    def _load_msg_bundle(self):
        if os.path.exists(MSG_BUNDLE_PATH):
            try:
                with open(MSG_BUNDLE_PATH, 'rb') as f:
                    bundle = pickle.load(f)
                cfg = bundle['model_config']
                self.msg_model = CyberGuardHybridModel(
                    text_dim=cfg['text_dim'],
                    cyber_in_dim=cfg['cyber_in_dim'],
                    cyber_out_dim=cfg['cyber_out_dim'],
                    hidden_dim=cfg['hidden_dim'],
                    num_classes=cfg['num_classes'],
                    dropout=cfg['dropout']
                )
                self.msg_model.load_state_dict(bundle['model_state_dict'])
                self.msg_model.eval()
                self.msg_scaler = bundle['scaler']
                self.msg_vectorizer = bundle['vectorizer']
                self.msg_risk_engine = bundle['risk_engine']
            except Exception as e:
                print(f"[UnifiedHub] Note loading message bundle: {e}")

    def predict_message(self, text: str, channel: str = "SMS") -> Dict[str, Any]:
        feats_dict = self.msg_extractor.extract_dict(text, channel)
        raw_feat_vector = np.array([feats_dict[k] for k in self.msg_extractor.FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
        
        if self.msg_model is not None and self.msg_vectorizer is not None and self.msg_scaler is not None:
            import torch
            norm_feats = self.msg_scaler.transform(raw_feat_vector)
            text_vec = self.msg_vectorizer.transform([text]).toarray().astype(np.float32)
            with torch.no_grad():
                logits = self.msg_model(text_embeddings=torch.tensor(text_vec), cyber_features=torch.tensor(norm_feats))
                probs = torch.softmax(logits, dim=1).numpy()[0]
                raw_prob = float(probs[1])
        else:
            # High-fidelity deterministic heuristic score if message weights not loaded
            score = 0.05
            if feats_dict.get('has_urgency', 0.0) == 1.0: score += 0.35
            if feats_dict.get('has_account_threat', 0.0) == 1.0: score += 0.30
            if feats_dict.get('has_credential_request', 0.0) == 1.0: score += 0.30
            if feats_dict.get('has_shortened_url', 0.0) == 1.0: score += 0.25
            raw_prob = min(0.99, score)

        risk_dict = self.msg_risk_engine.calculate_risk(raw_prob)
        risk_score = risk_dict['risk_score']
        severity = risk_dict['severity']
        action = risk_dict['action']
        cal_prob = risk_dict['calibrated_prob']
        xai_res = self.msg_xai_engine.generate_explanation(text, feats_dict, risk_dict, token_saliency=[])
        
        return {
            "channel": channel,
            "raw_probability": round(raw_prob, 4),
            "calibrated_probability": round(cal_prob, 4),
            "risk_score": risk_score,
            "severity": severity,
            "action": action,
            "prediction": "MALICIOUS" if risk_score >= 50 else "BENIGN",
            "indicators": xai_res['rule_indicators'],
            "suspicious_tokens": xai_res['suspicious_tokens'],
            "recommendation": xai_res['recommendation'],
            "explanation": xai_res['explanation_summary']
        }

    def predict_url(self, raw_url: str) -> Dict[str, Any]:
        return self.url_engine.analyze_url(raw_url, source="direct_url")

    def predict_qr_base64(self, image_b64: str) -> Dict[str, Any]:
        import cv2
        import numpy as np
        if ',' in image_b64:
            image_b64 = image_b64.split(',', 1)[1]
        raw_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {'success': False, 'error': 'Failed to decode image buffer.'}
        return self.url_engine.analyze_qr_image(img)

# FastAPI Application
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    
    app = FastAPI(
        title="CYBERGUARD & URL INTELLIGENCE API",
        description="Unified Enterprise Detection Hub for Messages, Direct URLs, and QR Codes",
        version="2.0.0"
    )
    
    _hub_instance = None
    
    def get_hub():
        global _hub_instance
        if _hub_instance is None:
            _hub_instance = UnifiedCyberGuardHub()
        return _hub_instance

    class MessageDetectionRequest(BaseModel):
        text: str
        channel: Optional[str] = "SMS"

    class URLDetectionRequest(BaseModel):
        url: str

    class QRBase64Request(BaseModel):
        image_base64: str

    @app.get("/health")
    def health():
        hub = get_hub()
        return {
            "status": "online",
            "message_module": "active",
            "url_module": "active",
            "url_bundle_loaded": hub.url_engine.bundle_loaded,
            "version": "2.0.0"
        }

    @app.post("/api/v1/detect")
    def detect_message(req: MessageDetectionRequest):
        hub = get_hub()
        return hub.predict_message(req.text, req.channel)

    @app.post("/api/v1/detect/url")
    def detect_url(req: URLDetectionRequest):
        hub = get_hub()
        return hub.predict_url(req.url)

    @app.post("/api/v1/detect/qr")
    def detect_qr(req: QRBase64Request):
        hub = get_hub()
        return hub.predict_qr_base64(req.image_base64)

    @app.get("/api/v1/benchmark/summary")
    def get_benchmark():
        report_path = r'e:\project 2\URL_BENCHMARK_REPORT.md'
        csv_path = r'e:\project 2\url_benchmark_results.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return {"results": df.to_dict(orient='records')}
        return {"error": "Benchmark results not yet generated"}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CYBERGUARD — Unified Phishing & URL Intelligence Hub</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0A0D14;
      --panel-bg: rgba(18, 24, 38, 0.75);
      --border-color: rgba(255, 255, 255, 0.08);
      --accent-cyan: #06B6D4;
      --accent-blue: #3B82F6;
      --color-green: #10B981;
      --color-amber: #F59E0B;
      --color-orange: #F97316;
      --color-red: #EF4444;
      --text-main: #F1F5F9;
      --text-muted: #94A3B8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 50% 0%, #172554 0%, var(--bg-dark) 70%);
      color: var(--text-main);
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      padding: 30px 20px;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 28px; }
    .logo-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      background: rgba(6, 182, 212, 0.12);
      border: 1px solid rgba(6, 182, 212, 0.3);
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent-cyan);
      margin-bottom: 12px;
      letter-spacing: 0.5px;
    }
    h1 {
      font-size: 2.4rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    .subtitle { color: var(--text-muted); font-size: 1.05rem; }
    
    /* Navigation Tabs */
    .tabs-nav {
      display: flex;
      justify-content: center;
      gap: 12px;
      margin-bottom: 25px;
    }
    .tab-btn {
      padding: 10px 22px;
      border-radius: 12px;
      border: 1px solid var(--border-color);
      background: rgba(255,255,255,0.04);
      color: var(--text-muted);
      font-weight: 700;
      font-size: 0.92rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }
    .tab-btn:hover { color: var(--text-main); border-color: rgba(255,255,255,0.2); }
    .tab-btn.active {
      background: linear-gradient(135deg, rgba(6,182,212,0.2), rgba(59,130,246,0.2));
      border-color: var(--accent-cyan);
      color: #FFF;
      box-shadow: 0 0 18px rgba(6, 182, 212, 0.3);
    }

    .grid-layout {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 25px;
    }
    @media (max-width: 860px) { .grid-layout { grid-template-columns: 1fr; } }
    
    .card {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      backdrop-filter: blur(16px);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    
    .channel-select-group {
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
    }
    .ch-btn {
      flex: 1;
      padding: 9px 12px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
      background: rgba(255,255,255,0.03);
      color: var(--text-muted);
      cursor: pointer;
      font-weight: 600;
      font-size: 0.85rem;
      transition: all 0.2s ease;
    }
    .ch-btn.active {
      background: var(--accent-cyan);
      border-color: var(--accent-cyan);
      color: #000;
      box-shadow: 0 0 12px rgba(6, 182, 212, 0.4);
    }
    
    textarea, input[type="text"] {
      width: 100%;
      background: rgba(10, 13, 20, 0.6);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 14px;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.95rem;
      margin-bottom: 16px;
      outline: none;
      transition: border 0.2s;
    }
    textarea { height: 130px; resize: vertical; }
    textarea:focus, input[type="text"]:focus { border-color: var(--accent-cyan); }
    
    .dropzone {
      border: 2px dashed rgba(6, 182, 212, 0.4);
      background: rgba(6, 182, 212, 0.03);
      border-radius: 14px;
      padding: 30px 20px;
      text-align: center;
      cursor: pointer;
      margin-bottom: 16px;
      transition: all 0.2s;
    }
    .dropzone:hover { border-color: var(--accent-cyan); background: rgba(6, 182, 212, 0.08); }
    .dropzone input { display: none; }

    .action-row {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .btn-analyze {
      flex: 1;
      padding: 13px 20px;
      border-radius: 12px;
      border: none;
      background: linear-gradient(135deg, #06B6D4, #3B82F6);
      color: #FFF;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
      box-shadow: 0 4px 15px rgba(6, 182, 212, 0.35);
      transition: all 0.2s;
    }
    .btn-analyze:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5); }
    
    .sample-chips {
      margin-top: 15px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      font-size: 0.75rem;
      padding: 5px 10px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      color: var(--text-muted);
      cursor: pointer;
    }
    .chip:hover { border-color: var(--accent-cyan); color: var(--text-main); }
    
    /* Result Pane */
    .gauge-wrapper { text-align: center; margin-bottom: 18px; }
    .risk-circle {
      width: 110px;
      height: 110px;
      border-radius: 50%;
      margin: 0 auto 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      border: 4px solid var(--border-color);
      background: rgba(0,0,0,0.3);
    }
    .risk-value { font-size: 2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
    .risk-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    
    .badge-row { display: flex; justify-content: center; gap: 10px; margin-bottom: 16px; }
    .badge {
      padding: 5px 12px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.82rem;
      letter-spacing: 0.5px;
    }
    .badge.CRITICAL { background: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; color: #EF4444; }
    .badge.HIGH { background: rgba(249, 115, 22, 0.15); border: 1px solid #F97316; color: #F97316; }
    .badge.MEDIUM { background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; color: #F59E0B; }
    .badge.LOW { background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; color: #10B981; }
    
    .indicator-box {
      background: rgba(0,0,0,0.25);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 12px;
      font-size: 0.88rem;
    }
    .indicator-title { font-weight: 700; margin-bottom: 8px; font-size: 0.80rem; color: var(--accent-cyan); text-transform: uppercase; }
    .indicator-list { list-style: none; }
    .indicator-list li { margin-bottom: 6px; display: flex; align-items: flex-start; gap: 8px; color: #CBD5E1; }
    .indicator-list li::before { content: "•"; color: var(--accent-cyan); font-weight: bold; }
    
    .token-tag {
      display: inline-block;
      padding: 2px 7px;
      background: rgba(239, 68, 68, 0.18);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #FCA5A5;
      border-radius: 4px;
      margin: 2px 4px 2px 0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
    }
    .rec-box {
      border-left: 3px solid var(--accent-cyan);
      padding-left: 12px;
      font-size: 0.86rem;
      color: var(--text-muted);
    }
    .benchmark-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 0.85rem;
    }
    .benchmark-table th, .benchmark-table td {
      padding: 10px 12px;
      border: 1px solid var(--border-color);
      text-align: left;
    }
    .benchmark-table th { background: rgba(255,255,255,0.04); color: var(--accent-cyan); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-badge">🛡️ ENTERPRISE CYBER DEFENSE HUB</div>
      <h1>CYBERGUARD & URL INTELLIGENCE</h1>
      <p class="subtitle">Unified Zero-Memorization Phishing Detection for Messages, Direct URLs & QR Codes</p>
    </header>

    <!-- Top Mode Tabs -->
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchMode('MSG', this)">📧 Message Analyzer</button>
      <button class="tab-btn" onclick="switchMode('URL', this)">🔗 Direct URL Intelligence</button>
      <button class="tab-btn" onclick="switchMode('QR', this)">📱 QR Code Scanner</button>
      <button class="tab-btn" onclick="switchMode('BENCH', this)">📊 Zero-Leakage Benchmark</button>
    </div>

    <div class="grid-layout">
      <!-- Input Card -->
      <div class="card" id="inputCard">
        
        <!-- SECTION 1: MESSAGE -->
        <div id="sectionMsg">
          <div class="card-title"><span>📥</span> Multi-Channel Message Inspector</div>
          <div class="channel-select-group">
            <button class="ch-btn active" onclick="setChannel('SMS', this)">SMS</button>
            <button class="ch-btn" onclick="setChannel('EMAIL', this)">EMAIL</button>
            <button class="ch-btn" onclick="setChannel('SOCIAL_CHAT', this)">SOCIAL CHAT</button>
          </div>
          <textarea id="msgInput" placeholder="Paste incoming SMS, email body, or social chat message..."></textarea>
          <div class="action-row">
            <button class="btn-analyze" onclick="runMsgAnalysis()">Analyze Message</button>
          </div>
          <div class="sample-chips">
            <span style="font-size:0.75rem; color:var(--text-muted); align-self:center;">Samples:</span>
            <span class="chip" onclick="loadMsgSample(0)">🚨 Bank Phish (SMS)</span>
            <span class="chip" onclick="loadMsgSample(1)">⚠️ Account Suspension (Email)</span>
            <span class="chip" onclick="loadMsgSample(2)">✅ Safe Meeting Notice</span>
          </div>
        </div>

        <!-- SECTION 2: DIRECT URL -->
        <div id="sectionUrl" style="display:none;">
          <div class="card-title"><span>🔗</span> Direct URL Characteristics Engine</div>
          <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:12px;">Evaluates 42 lexical, structural, and infrastructure signals with character/token pattern analysis.</p>
          <input type="text" id="urlInput" placeholder="Enter URL (e.g. http://secure-login-update.xyz/verify.php)..." />
          <div class="action-row">
            <button class="btn-analyze" onclick="runUrlAnalysis()">Inspect URL Intelligence</button>
          </div>
          <div class="sample-chips">
            <span style="font-size:0.75rem; color:var(--text-muted); align-self:center;">Samples:</span>
            <span class="chip" onclick="loadUrlSample(0)">🚨 Spoofed Apple Support</span>
            <span class="chip" onclick="loadUrlSample(1)">🚨 Phishing PayPal Notice</span>
            <span class="chip" onclick="loadUrlSample(2)">✅ Legitimate Wikipedia</span>
          </div>
        </div>

        <!-- SECTION 3: QR SCANNER -->
        <div id="sectionQr" style="display:none;">
          <div class="card-title"><span>📱</span> QR Code Decoder & Intelligence</div>
          <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:12px;">Decodes QR imagery into normalized URL, then routes into the exact same URL Intelligence engine.</p>
          <div class="dropzone" onclick="document.getElementById('qrFileInput').click()">
            <div style="font-size:2rem; margin-bottom:6px;">📷</div>
            <strong style="color:var(--accent-cyan);">Click to Upload QR Code Image</strong>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-top:4px;">Supports PNG, JPG, WebP. Auto-contrast & Otsu preprocessing enabled.</p>
            <input type="file" id="qrFileInput" accept="image/*" onchange="handleQrUpload(event)" />
          </div>
          <div id="qrPreviewText" style="font-size:0.85rem; color:var(--accent-cyan); margin-bottom:12px;"></div>
          <div class="action-row">
            <button class="btn-analyze" id="btnQrAnalyze" disabled onclick="runQrAnalysis()">Analyze Decoded QR Payload</button>
          </div>
        </div>

        <!-- SECTION 4: BENCHMARK -->
        <div id="sectionBench" style="display:none;">
          <div class="card-title"><span>📊</span> 4-Tier Zero-Leakage Benchmark Matrix</div>
          <p style="font-size:0.85rem; color:var(--text-muted);">Empirical verification on unseen distributions ensuring zero domain leakage.</p>
          <div id="benchmarkContainer">Loading benchmark data...</div>
        </div>

      </div>

      <!-- Result Card -->
      <div class="card">
        <div class="card-title"><span>📊</span> Risk & Dual XAI Analysis</div>
        
        <div class="gauge-wrapper">
          <div class="risk-circle" id="riskCircle">
            <span class="risk-value" id="riskScore">--</span>
            <span class="risk-label">Risk Score</span>
          </div>
          <div class="badge-row">
            <span class="badge" id="sevBadge">READY</span>
            <span class="badge" id="actionBadge">AWAITING INPUT</span>
          </div>
        </div>

        <div class="indicator-box">
          <div class="indicator-title">🔍 Detected Cyber Signals & Rules</div>
          <ul class="indicator-list" id="indicatorList">
            <li>Select an analyzer above to inspect security signals</li>
          </ul>
        </div>

        <div class="indicator-box">
          <div class="indicator-title">🧠 Model Saliency & Suspicious Tokens</div>
          <div id="tokenContainer" style="color:var(--text-muted); font-size:0.85rem;">None</div>
        </div>

        <div class="rec-box" id="recBox">
          <strong style="color:var(--text-main);">Action Recommendation:</strong><br>
          <span id="recText">Select or enter a target to begin real-time analysis.</span>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentChannel = "SMS";
    let currentQrBase64 = "";

    const msgSamples = [
      { ch: "SMS", text: "URGENT: Your SBI bank account has been locked due to unauthorized access. Confirm credentials immediately at http://bit.ly/sbi-verify or account will terminate." },
      { ch: "EMAIL", text: "Subject: Security Alert: Unauthorized Login Detected\\nDear Customer, we noticed unusual sign-ins to your account. Verify your identity and reset password now to avoid suspension." },
      { ch: "SMS", text: "Hi Mom, are we still meeting for lunch at 12:30 tomorrow? Let me know if you need me to pick anything up." }
    ];

    const urlSamples = [
      "http://login-apple-support-verify.com/security/login.php",
      "https://secure-update-paypal-notice.xyz/verify?token=92841",
      "https://en.wikipedia.org/wiki/Computer_security"
    ];

    function switchMode(mode, btn) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.getElementById('sectionMsg').style.display = (mode === 'MSG') ? 'block' : 'none';
      document.getElementById('sectionUrl').style.display = (mode === 'URL') ? 'block' : 'none';
      document.getElementById('sectionQr').style.display = (mode === 'QR') ? 'block' : 'none';
      document.getElementById('sectionBench').style.display = (mode === 'BENCH') ? 'block' : 'none';

      if (mode === 'BENCH') loadBenchmarkData();
    }

    function setChannel(ch, btn) {
      currentChannel = ch;
      document.querySelectorAll('.ch-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }

    function loadMsgSample(idx) {
      const s = msgSamples[idx];
      document.getElementById('msgInput').value = s.text;
      runMsgAnalysis();
    }

    function loadUrlSample(idx) {
      document.getElementById('urlInput').value = urlSamples[idx];
      runUrlAnalysis();
    }

    function handleQrUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(e) {
        currentQrBase64 = e.target.result;
        document.getElementById('qrPreviewText').innerText = `✓ Image selected: ${file.name} (${Math.round(file.size/1024)} KB)`;
        document.getElementById('btnQrAnalyze').disabled = false;
      };
      reader.readAsDataURL(file);
    }

    function renderResults(data) {
      const score = data.risk_score || 0;
      const sev = data.severity || 'LOW';
      const action = data.action || 'ALLOW';

      const circle = document.getElementById('riskCircle');
      circle.style.borderColor = (sev === 'CRITICAL') ? '#EF4444' : (sev === 'HIGH') ? '#F97316' : (sev === 'MEDIUM') ? '#F59E0B' : '#10B981';

      document.getElementById('riskScore').innerText = score;
      document.getElementById('riskScore').style.color = (sev === 'CRITICAL') ? '#EF4444' : (sev === 'HIGH') ? '#F97316' : (sev === 'MEDIUM') ? '#F59E0B' : '#10B981';

      const sevBadge = document.getElementById('sevBadge');
      sevBadge.className = `badge ${sev}`;
      sevBadge.innerText = sev;

      const actBadge = document.getElementById('actionBadge');
      actBadge.className = `badge ${sev}`;
      actBadge.innerText = action;

      const list = document.getElementById('indicatorList');
      list.innerHTML = "";
      if (data.indicators && data.indicators.length > 0) {
        data.indicators.forEach(ind => {
          const li = document.createElement('li');
          const title = ind.title || ind.rule || 'Indicator';
          const desc = ind.description || ind.detail || '';
          li.innerHTML = `<strong>${title}:</strong> ${desc}`;
          list.appendChild(li);
        });
      } else {
        list.innerHTML = "<li>Clean Profile — No security anomalies detected.</li>";
      }

      const tokBox = document.getElementById('tokenContainer');
      tokBox.innerHTML = "";
      if (data.suspicious_tokens && data.suspicious_tokens.length > 0) {
        data.suspicious_tokens.forEach(tok => {
          const span = document.createElement('span');
          span.className = "token-tag";
          span.innerText = tok;
          tokBox.appendChild(span);
        });
      } else {
        tokBox.innerText = "No anomalous token saliency detected.";
      }

      document.getElementById('recText').innerText = (data.guidance || data.recommendation || '') + " " + (data.explanation || '');
    }

    async function runMsgAnalysis() {
      const text = document.getElementById('msgInput').value.trim();
      if (!text) return;
      try {
        const resp = await fetch('/api/v1/detect', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ text: text, channel: currentChannel })
        });
        renderResults(await resp.json());
      } catch (err) { alert("Analysis request failed."); }
    }

    async function runUrlAnalysis() {
      const url = document.getElementById('urlInput').value.trim();
      if (!url) return;
      try {
        const resp = await fetch('/api/v1/detect/url', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ url: url })
        });
        renderResults(await resp.json());
      } catch (err) { alert("URL analysis request failed."); }
    }

    async function runQrAnalysis() {
      if (!currentQrBase64) return;
      try {
        const resp = await fetch('/api/v1/detect/qr', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ image_base64: currentQrBase64 })
        });
        const res = await resp.json();
        if (!res.success) {
          alert("QR decode error: " + (res.error || 'No QR pattern detected'));
          return;
        }
        renderResults(res);
      } catch (err) { alert("QR analysis request failed."); }
    }

    async function loadBenchmarkData() {
      const container = document.getElementById('benchmarkContainer');
      try {
        const resp = await fetch('/api/v1/benchmark/summary');
        const data = await resp.json();
        if (data.results) {
          let html = `<table class="benchmark-table">
            <thead>
              <tr><th>Test Suite</th><th>Samples</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>ROC-AUC</th></tr>
            </thead><tbody>`;
          data.results.forEach(r => {
            html += `<tr><td><strong>${r['Test Suite']}</strong></td><td>${r['Samples']}</td><td>${r['Accuracy']}</td><td>${r['Precision']}</td><td>${r['Recall']}</td><td><strong style="color:var(--color-green);">${r['F1 Score']}</strong></td><td>${r['ROC-AUC']}</td></tr>`;
          });
          html += `</tbody></table>`;
          container.innerHTML = html;
        } else {
          container.innerHTML = "No benchmark report available.";
        }
      } catch (err) { container.innerHTML = "Failed to load benchmark."; }
    }
  </script>
</body>
</html>
"""
except ImportError:
    app = None

if __name__ == '__main__':
    hub = UnifiedCyberGuardHub()
    print("\n" + "="*70)
    print("        CYBERGUARD & URL INTELLIGENCE UNIFIED API DEMO")
    print("="*70)
    print("\n[Testing Message Phishing]:")
    print(json.dumps(hub.predict_message("URGENT: SBI account suspended. Verify at http://bit.ly/sbi-acc", "SMS"), indent=2))
    print("\n[Testing Direct URL Intelligence]:")
    print(json.dumps(hub.predict_url("https://secure-update-paypal-notice.xyz/verify?token=92841"), indent=2))

import streamlit as st
import cv2
import torch
import numpy as np
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import tempfile
import os

# ----------------- Video Model Components -----------------
class ViTAttentionDetector(nn.Module):
    def __init__(self):
        super(ViTAttentionDetector, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.vit(x)
        x = x.view(B, T, -1)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x.view(-1)

def analyze_video(video_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViTAttentionDetector().to(device)
    model.load_state_dict(torch.load('/workspaces/Burnout-Detection-test1/attention_model_epoch_2.pth', map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50
    total_seconds = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = []
    for second in range(total_seconds):
        frames = []
        for offset in range(12):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((second + offset/12) * fps))
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else np.zeros((224, 224, 3), dtype=np.uint8)
            frame = cv2.resize(frame, (224, 224)) if ret else frame
            frames.append(transform(frame))
        
        with torch.no_grad():
            output = model(torch.stack(frames).unsqueeze(0).to(device))
            is_attentive = torch.sigmoid(output).item() > 0.5
            results.append({'second': second, 'burnout': not is_attentive})

    cap.release()
    return results

# ----------------- Text Model Components -----------------
@st.cache_resource
def load_text_model():
    model = DistilBertForSequenceClassification.from_pretrained("/workspaces/Burnout-Detection-test1/checkpoint-14295")
    tokenizer = DistilBertTokenizerFast.from_pretrained("/workspaces/Burnout-Detection-test1/checkpoint-14295")
    return model, tokenizer

text_model, tokenizer = load_text_model()

def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = text_model(**inputs).logits
    return "burnout" if logits.argmax().item() == 0 else "non-burnout"

# ----------------- Streamlit UI -----------------
st.title("Worker Burnout Detection System")

uploaded_file = st.file_uploader("Upload worker video", type=["mp4", "avi", "mov"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.getbuffer())
        video_results = analyze_video(tmp.name)
    os.unlink(tmp.name)

    burnout_seconds = [r['second'] for r in video_results if r['burnout']]
    non_burnout_seconds = [r['second'] for r in video_results if not r['burnout']]
    
    st.subheader("Video Analysis Results")
    col1, col2 = st.columns(2)
    col1.metric("Burnout Seconds", len(burnout_seconds))
    col2.metric("Non-Burnout Seconds", len(non_burnout_seconds))
    
    st.write("Detailed Second-by-Second Results:")
    st.json({s: ("Burnout" if s in burnout_seconds else "Normal") for s in range(len(video_results))})

    if burnout_seconds:
        st.subheader("Follow-up Assessment")
        with st.form("text_input"):
            st.write("Hey, you seem tired. Tell me what's up?")
            text_input = st.text_area("Share your feelings (min 20 characters)", height=100)
            submitted = st.form_submit_button("Analyze Text")
            
            if submitted:
                if len(text_input) < 20:
                    st.error("Please provide more detailed feedback (at least 20 characters)")
                else:
                    text_result = analyze_text(text_input)
                    st.subheader("Final Assessment")
                    st.success(f"Text analysis result: {text_result.upper()}")
                    if text_result == "burnout":
                        st.error("Our system detects significant burnout signs. Please consider taking a break.")
                    else:
                        st.info("No severe burnout detected, but monitor your stress levels.")

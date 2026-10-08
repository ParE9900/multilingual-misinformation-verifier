"""XLM-RoBERTa Natural Language Inference classifier module."""

from typing import Dict, Optional
from langdetect import detect


class NLIClassifier:
    """Cross-lingual NLI stance classifier."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 256):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load tokenizer and sequence classification model."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)

    def detect_language(self, text: str) -> str:
        """Detect language ISO code with fallback to en."""
        try:
            cleaned = text.strip() if text else ""
            if not cleaned:
                return "en"
            return detect(cleaned)
        except Exception:
            return "en"

    def predict_stance(self, claim: str, evidence: str) -> Dict[str, float]:
        """Predict stance probabilities by classifying the claim as real/fake."""
        if not self.model or not self.tokenizer:
            return {"supports": 0.5, "refutes": 0.5}
        
        import torch
        
        # Tokenize the claim
        inputs = self.tokenizer(
            claim, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            
        # Handle single-item batch squeeze
        if isinstance(probs, float):
            probs = [probs]
            
        # Label mapping from Phase 3: 0 = Real (supports), 1 = Fake (refutes)
        p_real = float(probs[0])
        p_fake = float(probs[1])
        
        return {"supports": p_real, "refutes": p_fake}
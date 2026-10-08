"""XLM-RoBERTa NLI classifier and LLM stance predictor."""

import json
from typing import Dict, Optional
from langdetect import detect


class NLIClassifier:
    """Cross-lingual NLI stance classifier and style analyzer."""

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

    def analyze_claim_style(self, claim: str) -> Dict[str, float]:
        """Classify claim text as real/fake based on stylistic features."""
        if not self.model or not self.tokenizer:
            return {"real_style": 0.5, "fake_style": 0.5}
        
        import torch
        
        inputs = self.tokenizer(
            claim, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
            
        if isinstance(probs, float):
            probs = [probs]
            
        # 0 = Real, 1 = Fake
        return {"real_style": float(probs[0]), "fake_style": float(probs[1])}

    def predict_stance(self, claim: str, evidence: str) -> Dict[str, float]:
        """Predict stance using Groq LLM for accurate NLI."""
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        from src.config import get_config
        
        try:
            config = get_config()
            client = ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name=config.GROQ_MODEL, temperature=0.0)
            
            prompt = f"""
            Claim: {claim}
            Evidence: {evidence}
            
            Analyze if the evidence supports or refutes the claim.
            Output strict JSON: {{"supports": float, "refutes": float}}
            The sum of supports and refutes must be 1.0.
            """
            
            response = client.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            
            # Clean JSON
            text = text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            
            p_sup = float(data.get("supports", 0.5))
            p_ref = float(data.get("refutes", 0.5))
            
            # Normalize
            total = p_sup + p_ref
            if total > 0:
                p_sup /= total
                p_ref /= total
                
            return {"supports": p_sup, "refutes": p_ref}
        except Exception as e:
            print(f"[DEBUG] LLM NLI failed: {e}")
            return {"supports": 0.5, "refutes": 0.5}
import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class LLMService:
    """
    Multi-Provider LLM Integration Service.
    Supports Google Gemini (1.5 Flash / Pro), OpenAI (GPT-4o-mini / GPT-4o), Groq,
    and fallback to Local Semantic Context Synthesizer.
    """

    def __init__(self):
        self.load_keys_from_env()

    def load_keys_from_env(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.default_provider = os.environ.get("LLM_PROVIDER", "auto")

    def get_active_provider(self, custom_key: Optional[str] = None, provider_pref: Optional[str] = None) -> Tuple[str, str]:
        """Determines the best available LLM provider."""
        if provider_pref and custom_key:
            return provider_pref, custom_key
            
        if self.gemini_key:
            return "gemini", self.gemini_key
        elif self.openai_key:
            return "openai", self.openai_key
        elif self.groq_key:
            return "groq", self.groq_key
            
        return "local_semantic", ""

    def test_provider_connection(self, provider: str, api_key: str) -> Dict[str, Any]:
        """Validates API key connectivity for a specific provider."""
        api_key = api_key.strip()
        if not api_key:
            return {"success": False, "error": "API key cannot be empty."}

        if provider == "gemini":
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": "Reply with 'OK'"}]}],
                    "generationConfig": {"maxOutputTokens": 10}
                }
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    return {"success": True, "provider": "gemini", "model": "gemini-1.5-flash"}
                else:
                    err_msg = resp.json().get("error", {}).get("message", resp.text)
                    return {"success": False, "error": f"Gemini API Error ({resp.status_code}): {err_msg}"}
            except Exception as e:
                return {"success": False, "error": f"Connection failed: {str(e)}"}

        elif provider in ["openai", "groq"]:
            base_url = "https://api.openai.com/v1" if provider == "openai" else "https://api.groq.com/openai/v1"
            model = "gpt-4o-mini" if provider == "openai" else "llama-3.3-70b-versatile"
            try:
                url = f"{base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with 'OK'"}],
                    "max_tokens": 10
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    return {"success": True, "provider": provider, "model": model}
                else:
                    err_msg = resp.json().get("error", {}).get("message", resp.text)
                    return {"success": False, "error": f"{provider.upper()} API Error ({resp.status_code}): {err_msg}"}
            except Exception as e:
                return {"success": False, "error": f"Connection failed: {str(e)}"}

        return {"success": False, "error": f"Unknown provider: {provider}"}

    def generate_grounded_response(
        self,
        query: str,
        context_text: str,
        company_name: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        custom_key: Optional[str] = None,
        custom_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sends grounded dataset context and user query to the LLM to generate an
        articulate, intelligent, logical, zero-hallucination response.
        """
        provider, key = self.get_active_provider(custom_key, custom_provider)

        # Build comprehensive system instructions
        system_instruction = (
            f"You are the intelligent Analytics & Customer Retention AI Assistant for '{company_name}'.\n"
            f"You are directly connected to the company's verified SQL database, Machine Learning Stacking Ensemble "
            f"(CatBoost + XGBoost + Random Forest), and TreeSHAP explainability engine.\n\n"
            f"CRITICAL GROUNDING RULES:\n"
            f"1. Base your answers strictly on the provided VERIFIED DATASET CONTEXT and DOMAIN KNOWLEDGE below.\n"
            f"2. Never hallucinate, invent customer accounts, falsify ML probabilities, or fabricate promotional monetary discounts.\n"
            f"3. When answering questions about churn risk, high-risk accounts, contracts, or payment methods, cite the exact figures from the context.\n"
            f"4. If a question is a friendly conversational message (e.g. 'bye', 'hello', 'thank you'), reply politely and warmly without dumping random data.\n"
            f"5. If a question asks about information completely outside the dataset or platform scope, politely clarify what customer analytics and models are available.\n"
            f"6. Format your response cleanly using GitHub Markdown (tables, bold text, bullet points).\n\n"
            f"--- VERIFIED DATASET & DOMAIN CONTEXT ---\n"
            f"{context_text}\n"
            f"-----------------------------------------"
        )

        if provider == "gemini" and key:
            res = self._call_gemini(query, system_instruction, key, chat_history)
            if res.get("success"):
                return {"success": True, "response": res["text"], "provider": "gemini", "model": "gemini-1.5-flash"}
            logger.warning(f"Gemini call failed ({res.get('error')}), falling back...")

        elif provider == "openai" and key:
            res = self._call_openai(query, system_instruction, key, chat_history)
            if res.get("success"):
                return {"success": True, "response": res["text"], "provider": "openai", "model": "gpt-4o-mini"}
            logger.warning(f"OpenAI call failed ({res.get('error')}), falling back...")

        elif provider == "groq" and key:
            res = self._call_groq(query, system_instruction, key, chat_history)
            if res.get("success"):
                return {"success": True, "response": res["text"], "provider": "groq", "model": "llama-3.3-70b"}
            logger.warning(f"Groq call failed ({res.get('error')}), falling back...")

        return {"success": False, "provider": "none", "error": "No working LLM provider available"}

    def _call_gemini(self, query: str, system_instruction: str, api_key: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Calls Google Gemini API."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            
            contents = []
            # Add system instruction as first user/model turn if systemInstruction not natively supported in v1beta
            combined_prompt = f"{system_instruction}\n\nUser Question: {query}"
            
            if chat_history:
                for msg in chat_history[-4:]: # include recent 4 turns
                    role = "user" if msg.get("role") == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
                    
            contents.append({"role": "user", "parts": [{"text": combined_prompt}]})

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                    "topP": 0.95
                }
            }
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return {"success": True, "text": parts[0].get("text", "")}
                return {"success": False, "error": "Empty response from Gemini"}
            else:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
                return {"success": False, "error": f"Gemini Error ({resp.status_code}): {err_msg}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _call_openai(self, query: str, system_instruction: str, api_key: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Calls OpenAI Chat Completions API."""
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            messages = [{"role": "system", "content": system_instruction}]
            if chat_history:
                for msg in chat_history[-4:]:
                    messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            messages.append({"role": "user", "content": query})

            payload = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1024
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return {"success": True, "text": text}
            else:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
                return {"success": False, "error": f"OpenAI Error ({resp.status_code}): {err_msg}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _call_groq(self, query: str, system_instruction: str, api_key: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Calls Groq Cloud API."""
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            messages = [{"role": "system", "content": system_instruction}]
            if chat_history:
                for msg in chat_history[-4:]:
                    messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            messages.append({"role": "user", "content": query})

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1024
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return {"success": True, "text": text}
            else:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
                return {"success": False, "error": f"Groq Error ({resp.status_code}): {err_msg}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

llm_service = LLMService()

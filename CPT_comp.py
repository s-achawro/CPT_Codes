import os
import re
import csv
import math
import json
import base64
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional

#import pandas as pd
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

# Your Azure OpenAI settings
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "https://hcpt-code-resource.openai.azure.com/")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-5-nano")
API_VERSION = "2025-01-01-preview"

# Your local data files
ANESTHESIA_FILE = os.getenv("ANESTHESIA_FILE", "/mnt/data/Y2024_01) Anesthesia (00000 - 09999).xlsx")
# Optional codebook (hcpcs,description)
CODEBOOK_CSV = os.getenv("CODEBOOK_CSV", "anesthesia_codebook.csv")  # put your own path here

# Retrieval parameters
TOP_K = 8  # how many candidates to pass into the model for final selection


# ------------------------------------------------------------------------------
# UTILS: text cleaning and scoring (simple keyword/BM25-lite style)
# ------------------------------------------------------------------------------

def normalize_text(t: str) -> str:
    t = t.lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(t: str) -> List[str]:
    return normalize_text(t).split()

def score(query: str, doc: str, k1: float = 1.6, b: float = 0.75, avgdl: float = 40.0) -> float:
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    if not d_tokens:
        return 0.0
    tf = Counter(d_tokens)
    dl = len(d_tokens)
    s = 0.0
    for t in set(q_tokens):
        f = tf.get(t, 0)
        if f == 0:
            continue
        idf = 1.5
        denom = f + k1 * (1 - b + b * (dl / (avgdl if avgdl > 0 else 1)))
        s += idf * ((f * (k1 + 1)) / denom)
    return s

def pick_candidates_from_codebook(user_desc: str, codebook: Dict[str, str], k: int = 8) -> List[Tuple[str, str, float]]:
    scored = []
    for code, desc in codebook.items():
        s = score(user_desc, f"{code} {desc}")
        scored.append((code, desc, s))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:max(1, k)]



# ------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------

def pdf_to_text(pdf_path: str) -> str:
    text = ""
    try:
        import pdfplumber  # pip install pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                text += "\n"
        if text.strip():
            return text
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader  # pip install PyPDF2
        r = PdfReader(pdf_path)
        for p in r.pages:
            text += p.extract_text() or ""
            text += "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {e}")

def parse_anesthesia_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Parse lines like:
      00100 5 Anesthesia for procedures on salivary glands, including biopsy
    Handle wrapped descriptions by appending following lines that don't start with a new code.
    """
    raw = pdf_to_text(pdf_path)
    lines = [ln.strip() for ln in raw.splitlines()]

    # Skip obvious headers/footers
    skip_if_contains = [
        "Anesthesia Service Codes Spreadsheet",
        "Procedure codes and base units are obtained from the Centers for Medicare & Medicaid Services",
        "Code Units Description",
    ]

    code_desc: Dict[str, str] = {}
    cur_code = None
    cur_desc = []

    code_line = re.compile(r"^(\d{5})\s+(\d+)\s+(.*)$")

    def flush():
        nonlocal cur_code, cur_desc
        if cur_code and cur_desc:
            # collapse whitespace, keep punctuation
            desc = " ".join(" ".join(cur_desc).split())
            code_desc[cur_code] = desc
        cur_code, cur_desc = None, []

    for ln in lines:
        if not ln or any(s in ln for s in skip_if_contains):
            continue

        m = code_line.match(ln)
        if m:
            # start of a new code row
            flush()
            cur_code = m.group(1)
            # units = m.group(2)  # available if needed later
            first_chunk = m.group(3).strip()
            cur_desc = [first_chunk] if first_chunk else []
        else:
            # continuation line (wrapped description)
            if cur_code:
                cur_desc.append(ln)

    flush()
    return code_desc





def load_codebook(path: str) -> Dict[str, str]:
    """
    Load a codebook CSV with columns: hcpcs,description
    Returns dict: { '00100': 'Anesthesia for ...', ... }
    """
    if not os.path.exists(path):
        return {}
    mapping = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row.get("hcpcs", "")).strip()
            desc = str(row.get("description", "")).strip()
            if code:
                mapping[code] = desc
    return mapping


# ------------------------------------------------------------------------------
# RETRIEVAL: pick top-K candidate codes to pass to the model
# ------------------------------------------------------------------------------

def pick_candidates(user_description: str, codes_in_file: List[str], codebook: Dict[str, str], k: int = TOP_K) -> List[Tuple[str, str, float]]:
    """
    Returns a list of (code, description, score) top candidates.
    If we don't have a description for a code, we use an empty string but still include the code.
    """
    scored: List[Tuple[str, str, float]] = []
    # If codebook is present, score against descriptions; else score by code token overlap (weak but still deterministic)
    for code in codes_in_file:
        desc = codebook.get(code, "")
        corpus = f"{code} {desc}" if desc else code
        score = score(user_description, corpus)  
        scored.append((code, desc, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[: max(1, k)]


# ------------------------------------------------------------------------------
# AZURE OPENAI CLIENT
# ------------------------------------------------------------------------------

def get_azure_client() -> AzureOpenAI:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT_URL,
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION,
    )
    return client


# ------------------------------------------------------------------------------
# MODEL CALL: ask the model to choose the best code among candidates
# ------------------------------------------------------------------------------

def choose_code_with_model(client: AzureOpenAI, user_description: str, candidates: List[Tuple[str, str, float]]) -> Dict:
    """
    Sends only a small candidate set to the model and asks it to pick exactly one code.
    The model never sees your whole dataset—only the short list we retrieved locally.
    """
    # Build a compact table for the prompt
    rows = []
    for code, desc, score in candidates:
        rows.append({"hcpcs": code, "description": desc or "(no description available)", "retrieval_score": round(score, 3)})

    system_msg = {
        "role": "system",
        "content": (
            "You are a medical coding assistant. You must choose the single most appropriate HCPCS/CPT anesthesia code "
            "from the provided candidate list strictly based on the user description and the candidate descriptions. "
            "If no description is provided for a candidate, prefer candidates whose descriptions clearly match the anatomy/procedure."
        )
    }

    # We instruct the model to answer in JSON only
    user_msg = {
        "role": "user",
        "content": (
            "Task: Given the anesthesia procedure description below, pick ONE code from candidates.\n\n"
            f"Description: {user_description}\n\n"
            "Candidates (JSON array):\n"
            + json.dumps(rows, ensure_ascii=False, indent=2)
            + "\n\nReturn a JSON object with fields:\n"
              "  code: the chosen HCPCS/CPT code,\n"
              "  rationale: brief reason for the choice (1-2 sentences).\n"
              "If you cannot safely decide, return the most likely code and note the uncertainty."
        )
    }

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[system_msg, user_msg],
        temperature=0.2,
        max_completion_tokens=400
    )

    # Parse JSON from the model’s reply (be tolerant)
    text = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(text)
    except Exception:
        # If the model didn't return valid JSON, wrap it
        parsed = {"code": None, "rationale": f"Non-JSON model reply: {text}"}
    return parsed


# ------------------------------------------------------------------------------
# PUBLIC FUNCTION: classify an anesthesia description
# ------------------------------------------------------------------------------

class AnesthesiaPDFCoder:
    def __init__(self, pdf_path: str, top_k: int = 8):
        self.pdf_path = pdf_path
        self.codebook: Dict[str, str] = parse_anesthesia_pdf(pdf_path)
        if not self.codebook:
            raise ValueError("No codes found in the provided PDF.")
        self.top_k = top_k
        self.client = get_azure_client()

    def classify(self, description: str) -> Dict:
        candidates = pick_candidates_from_codebook(description, self.codebook, k=self.top_k)
        rows = [{"hcpcs": c, "description": d, "retrieval_score": round(s, 3)} for c, d, s in candidates]

        system_msg = {
            "role": "system",
            "content": (
                "You are a medical coding assistant. Choose the single most appropriate HCPCS/CPT anesthesia code "
                "from the candidate list strictly based on the user description and candidate descriptions."
            )
        }
        user_msg = {
            "role": "user",
            "content": (
                "Task: Given the anesthesia procedure description below, pick ONE code from candidates.\n\n"
                f"Description: {description}\n\n"
                "Candidates (JSON array):\n"
                + json.dumps(rows, ensure_ascii=False, indent=2)
                + "\n\nReturn a JSON object with fields:\n"
                  "  code: the chosen code,\n"
                  "  rationale: brief reason (1–2 sentences)."
            )
        }

        resp = self.client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[system_msg, user_msg],
            temperature=0.2,
            max_completion_tokens=300
        )

        text = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"code": None, "rationale": f"Non-JSON reply: {text}"}

        return {
            "query": description,
            "code": parsed.get("code"),
            "rationale": parsed.get("rationale"),
            "candidates_considered": rows
        }



# ------------------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    PDF_PATH = os.getenv("ANES_PDF", "/mnt/data/anesthesia-service-codes-spreadsheet-101-cmr-316.pdf")
    coder = AnesthesiaPDFCoder(PDF_PATH, top_k=8)

    user_query = "Anesthesia for transurethral resection of the prostate"
    result = coder.classify(user_query)
    print(json.dumps(result, indent=2))

import os
import json
import httpx
import logging
import fitz  # PyMuPDF
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fidelity OS - Multi-Agent Engine")

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

NVIDIA_API_KEY = "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"
SUPABASE_URL = "https://wzfduiwirpythpndrdtm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind6ZmR1aXdpcnB5dGhwbmRyZHRtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4NDUzMjA2OSwiZXhwIjoyMTAwMTA4MDY5fQ.uUZ8smKB0iF5jxzWwzJI0bCR-DXq8MoKRIGESgNiusA"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# --- MULTI-AGENT ORCHESTRATOR ---
async def call_nvidia_nim(model: str, system_prompt: str, user_prompt: str) -> dict:
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.2, "top_p": 0.2, "max_tokens": 2048, "stream": False}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(NVIDIA_BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)

# Agent A: Evidence Graph Extractor
async def agent_a_extract_resume(resume_text: str) -> dict:
    system_prompt = """You are an elite Recruitment Analyst AI. Extract verifiable evidence from a candidate's resume. Ignore marketing fluff. Extract ALL programming languages, frameworks, and tools into "tech_stack". 
    Output STRICTLY JSON: {"identity": {"name": "", "email": "", "links": []}, "experience": [{"company": "", "role": "", "bullets": [], "tech_stack": [], "metrics": [], "tradeoffs": []}]}"""
    return await call_nvidia_nim("z-ai/glm-5.2", system_prompt, resume_text)

# Agent B: JD Intelligence
async def agent_b_parse_jd(jd_text: str) -> dict:
    system_prompt = """You are a Job Description parsing agent. Extract technical requirements and flags. ONLY extract specific programming languages, frameworks, tools. DO NOT extract soft skills.
    Output STRICTLY JSON: {"domain": "SWE", "reqs": ["Python"], "sponsorship": true, "management": false}"""
    return await call_nvidia_nim("z-ai/glm-5.2", system_prompt, jd_text)

# Agent E: The Anti-Homogenization Engine (Set-Theory Formatter)
async def agent_e_anti_homogenize(original_bullet: str, target_domain: str, jd_reqs: list) -> dict:
    system_prompt = f"""You are an ATS optimization engine. Rewrite the user's bullet to match the exact string requirements for a {target_domain} role.
    To avoid AI homogenization, you MUST randomly vary the sentence structure using one of three formats: 
    1. [Action] + [Context] + [Tradeoff] + [Result]
    2. [Context] + [Result] + [Action] + [Tradeoff]
    3. [Tradeoff] + [Action] + [Result] + [Context]
    DO NOT lie or invent metrics. Only rephrase for string match.
    Output STRICTLY JSON: {{"formatted_bullet": ""}}"""
    user_prompt = f"Original Bullet: {original_bullet}\nTarget Reqs: {', '.join(jd_reqs)}"
    return await call_nvidia_nim("mistralai/mixtral-8x7b-instruct-v0.1", system_prompt, user_prompt)

# Agent C2: Inference Evaluator
async def agent_c2_inference(candidate_skills: list, missing_skills: list) -> dict:
    system_prompt = """You are a strict Technical Evaluator. Determine if the candidate's verified skills make it a REASONABLE INDUSTRY INFERENCE that they possess the missing skill.
    DO NOT infer across different core languages. Output STRICTLY JSON: {"inferred_bypasses": [{"skill": "Pandas", "reason": "Inferred from Python"}], "truly_missing": ["C++"]}"""
    user_prompt = f"Candidate Skills: {', '.join(candidate_skills)}\nMissing Skills: {', '.join(missing_skills)}"
    return await call_nvidia_nim("meta/llama-3.1-8b-instruct", system_prompt, user_prompt)

def agent_c_initial_match(candidate_skills: list, jd_reqs: list) -> dict:
    # Same as before, returns passed/bypasses/missing
    bypasses, missing = [], []
    c_skills_lower = [str(s).lower() for s in candidate_skills]
    for req in jd_reqs:
        req_lower = str(req).lower()
        if req_lower in c_skills_lower or any(req_lower in s or s in req_lower for s in c_skills_lower): continue
        missing.append(req)
    return {"passed": len(missing) == 0, "bypasses": bypasses, "missing": missing}

def extract_text_from_pdf(file: bytes) -> str:
    doc = fitz.open(stream=file, filetype="pdf")
    return "".join(page.get_text("text") for page in doc)

# --- ENDPOINTS ---

@app.post("/onboard/")
async def onboard_candidate(file: UploadFile = File(...)):
    file_bytes = await file.read()
    resume_text = extract_text_from_pdf(file_bytes) if file.filename.endswith(".pdf") else file_bytes.decode("utf-8")
    evidence_data = await agent_a_extract_resume(resume_text)
    candidate_row = supabase.table("candidates").insert({"target_domain": "SWE"}).execute()
    candidate_id = str(candidate_row.data[0]["id"])
    for exp in evidence_data.get("experience", []):
        supabase.table("evidence").insert({"candidate_id": candidate_id, "raw_bullet": " | ".join(exp.get("bullets", [])), "extracted_skills": exp.get("tech_stack", [])}).execute()
    return {"status": "Success", "candidate_id": candidate_id}

@app.post("/evaluate-application/")
async def evaluate_application(candidate_id: str, jd_text: str):
    jd_data = await agent_b_parse_jd(jd_text)
    if jd_data.get("sponsorship") == False:
        return {"status": "VETOED", "reason": "Company does not provide sponsorship."}
    
    db_res = supabase.table("evidence").select("extracted_skills, raw_bullet").eq("candidate_id", candidate_id).execute()
    if not db_res.data: return {"status": "VETOED", "reason": "Candidate ID not found."}
    
    c_skills = list(set(s for row in db_res.data for s in row.get("extracted_skills", [])))
    c_bullets = [row.get("raw_bullet", "") for row in db_res.data if row.get("raw_bullet")]
    
    initial_match = agent_c_initial_match(c_skills, jd_data.get("reqs", []))
    final_bypasses, truly_missing = initial_match.get("bypasses", []), initial_match.get("missing", [])
    
    if truly_missing:
        inf_res = await agent_c2_inference(c_skills, truly_missing)
        final_bypasses.extend([f"Inferred: {b['skill']}" for b in inf_res.get("inferred_bypasses", [])])
        truly_missing = inf_res.get("truly_missing", [])
        
    if truly_missing:
        return {"status": "VETOED", "reason": f"Missing: {truly_missing}"}
    
    formatted_data = await agent_e_anti_homogenize(c_bullets[0] if c_bullets else "", jd_data.get("domain", "SWE"), jd_data.get("reqs", []))
    backdoor_dm = f"Hi Hiring Manager, I saw your req for {jd_data.get('domain', 'SWE')}. {formatted_data.get('formatted_bullet', '')} I'd love to chat."
    
    return {"status": "READY_FOR_SUBMISSION", "ats_optimized_bullet": formatted_data.get("formatted_bullet", ""), "linkedin_dm_draft": backdoor_dm}

# THE AUTONOMOUS RADAR (Scrapes direct career pages)
@app.post("/radar/scan/")
async def radar_scan(company_url: str):
    # In production, this uses stealth proxies. For MVP, simple httpx fetch.
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    async with httpx.AsyncClient() as client:
        res = await client.get(company_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        # Basic extraction of job links (simplified for deployment)
        jobs = [{"title": tag.text.strip(), "url": tag['href']} for tag in soup.find_all('a', href=True) if 'job' in tag.text.lower() or 'job' in tag['href'].lower()]
        return {"status": "Radar Active", "jobs_found": len(jobs), "jobs": jobs[:5]}

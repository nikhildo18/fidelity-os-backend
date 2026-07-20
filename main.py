import os
import json
import httpx
import logging
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fidelity OS - Multi-Agent Engine")

# --- CORS POLICY ---
origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENVIRONMENT CONFIG ---
NVIDIA_API_KEY = "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"
SUPABASE_URL = "https://wzfduiwirpythpndrdtm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind6ZmR1aXdpcnB5dGhwbmRyZHRtIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4NDUzMjA2OSwiZXhwIjoyMTAwMTA4MDY5fQ.uUZ8smKB0iF5jxzWwzJI0bCR-DXq8MoKRIGESgNiusA"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# --- MULTI-AGENT LLM ORCHESTRATOR ---

async def call_nvidia_nim(model: str, system_prompt: str, user_prompt: str) -> dict:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "top_p": 0.1,
        "max_tokens": 2048,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(NVIDIA_BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            raise Exception(f"LLM failed to parse: {str(e)}")

# Agent A: Evidence Graph Extractor
async def agent_a_extract_resume(resume_text: str) -> dict:
    system_prompt = """You are an elite Recruitment Analyst AI. Your job is to extract verifiable evidence from a candidate's resume. 
    You must ignore marketing fluff ("results-driven", "passionate") and extract ONLY hard facts.
    You MUST extract ALL programming languages, frameworks, libraries, and tools into the "tech_stack" array.
    Extract links, bullet points that contain metrics or architectural decisions.
    If a metric is missing, output null. Do NOT invent data.
    
    Output STRICTLY in this JSON format:
    {
      "identity": {
        "name": "",
        "email": "",
        "links": []
      },
      "experience": [
        {
          "company": "",
          "role": "",
          "bullets": ["", ""],
          "tech_stack": ["Python", "Pandas", "AWS"],
          "metrics": []
        }
      ]
    }"""
    return await call_nvidia_nim("z-ai/glm-5.2", system_prompt, resume_text)

# Agent B: JD Intelligence
async def agent_b_parse_jd(jd_text: str) -> dict:
    system_prompt = """You are a Job Description parsing agent. Extract the technical requirements and flags. 
    ONLY extract specific programming languages, frameworks, tools, and software (e.g., Python, Kubernetes, AWS, Pandas).
    DO NOT extract job titles, years of experience, soft skills, or business concepts like "scalable data pipelines".
    Output STRICTLY in this JSON format: 
    {"domain": "SWE", "reqs": ["Python", "Kubernetes"], "sponsorship": false, "management": false}"""
    return await call_nvidia_nim("z-ai/glm-5.2", system_prompt, jd_text)

# Agent C2: The Inference Evaluator
async def agent_c2_inference(candidate_skills: list, missing_skills: list) -> dict:
    system_prompt = """You are a strict, intellectual Technical Evaluator. 
    The candidate is missing some skills requested in the Job Description. 
    Your job is to determine if the candidate's verified skills make it a REASONABLE INDUSTRY INFERENCE that they possess or can immediately perform the missing skill.
    (e.g., If candidate has Python + NumPy, inferring Pandas is reasonable. Inferring C++ is NOT reasonable).
    DO NOT infer across different core languages or vastly different paradigms.
    
    Output STRICTLY in this JSON format:
    {
      "inferred_bypasses": [
        {"skill": "Pandas", "reason": "Inferred from candidate's Python and Data Analysis experience"}
      ],
      "truly_missing": ["C++"]
    }"""
    user_prompt = f"Candidate Skills: {', '.join(candidate_skills)}\nMissing Skills: {', '.join(missing_skills)}"
    return await call_nvidia_nim("meta/llama-3.1-8b-instruct", system_prompt, user_prompt)

# Agent D: Domain Formatter
async def agent_d_format_bullet(original_bullet: str, target_domain: str, jd_reqs: list) -> dict:
    system_prompt = f"""You are an ATS optimization engine. Rewrite the user's bullet to match the exact string requirements for a {target_domain} role. 
    DO NOT lie or invent metrics. Only rephrase for string match.
    Output STRICTLY in this JSON format: {{"formatted_bullet": ""}}"""
    user_prompt = f"Original Bullet: {original_bullet}\nTarget Reqs: {', '.join(jd_reqs)}"
    return await call_nvidia_nim("mistralai/mixtral-8x7b-instruct-v0.1", system_prompt, user_prompt)

# Agent C: The Deterministic Veto (Initial String Match)
SEMANTIC_EQUIVALENCE_MAP = {
    "Kubernetes": ["Docker Swarm", "Nomad"],
    "PyTorch": ["TensorFlow", "JAX"]
}

def agent_c_initial_match(candidate_skills: list, jd_reqs: list) -> dict:
    bypasses = []
    missing = []
    c_skills_lower = [str(s).lower() for s in candidate_skills]
    
    for req in jd_reqs:
        req_lower = str(req).lower()
        if req_lower in c_skills_lower:
            continue
        if any(req_lower in skill or skill in req_lower for skill in c_skills_lower):
            continue
        if req in SEMANTIC_EQUIVALENCE_MAP:
            if any(equiv.lower() in c_skills_lower for equiv in SEMANTIC_EQUIVALENCE_MAP[req]):
                bypasses.append(f"Bypassed: Mapped equivalent to {req}")
                continue
        missing.append(req)
        
    return {"passed": len(missing) == 0, "bypasses": bypasses, "missing": missing}

# --- FILE PARSING ---
def extract_text_from_pdf(file: bytes) -> str:
    try:
        doc = fitz.open(stream=file, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        raise Exception("Failed to read PDF file.")

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "Fidelity OS Backend is Alive"}

@app.post("/onboard/")
async def onboard_candidate(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if file.filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(file_bytes)
        else:
            resume_text = file_bytes.decode("utf-8")
            
        evidence_data = await agent_a_extract_resume(resume_text)
        
        candidate_row = supabase.table("candidates").insert({
            "target_domain": "SWE"
        }).execute()
        candidate_id = str(candidate_row.data[0]["id"])
        
        for exp in evidence_data.get("experience", []):
            supabase.table("evidence").insert({
                "candidate_id": candidate_id,
                "raw_bullet": " | ".join(exp.get("bullets", [])),
                "extracted_skills": exp.get("tech_stack", []),
                "extracted_metrics": exp.get("metrics", []),
                "is_verified": True 
            }).execute()
            
        return {"status": "Evidence Graph built", "candidate_id": candidate_id, "data": evidence_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-application/")
async def evaluate_application(candidate_id: str, jd_text: str):
    try:
        jd_data = await agent_b_parse_jd(jd_text)
        
        if jd_data.get("sponsorship") == False:
            return {"status": "VETOED", "reason": "Company does not provide sponsorship."}
            
        db_response = supabase.table("evidence").select("extracted_skills, raw_bullet").eq("candidate_id", candidate_id).execute()
        
        if not db_response.data:
            return {"status": "VETOED", "reason": "Candidate ID not found in database."}
            
        candidate_skills = list(set(skill for row in db_response.data for skill in row.get("extracted_skills", [])))
        candidate_bullets = [row.get("raw_bullet", "") for row in db_response.data if row.get("raw_bullet")]
        
        logger.info(f"Candidate Skills: {candidate_skills}")
        logger.info(f"JD Reqs: {jd_data.get('reqs', [])}")
        
        # 1. Agent C: Initial Boolean/String Match
        initial_match = agent_c_initial_match(candidate_skills, jd_data.get("reqs", []))
        
        final_bypasses = initial_match.get("bypasses", [])
        truly_missing = initial_match.get("missing", [])
        
        # 2. Agent C2: Intellectual Inference (If initial match missed some skills)
        if truly_missing:
            inference_result = await agent_c2_inference(candidate_skills, truly_missing)
            
            # Add inferred skills to bypasses
            for bypass in inference_result.get("inferred_bypasses", []):
                final_bypasses.append(f"Inferred Bypass: {bypass['skill']} ({bypass['reason']})")
            
            truly_missing = inference_result.get("truly_missing", [])
            
        # 3. Final Veto Check
        if truly_missing:
            return {
                "status": "VETOED", 
                "missing": truly_missing,
                "reason": f"Hard constraint violation. Missing: {truly_missing}"
            }
            
        # 4. Agent D: Format Application
        original_bullet = candidate_bullets[0] if candidate_bullets else ""
        formatted_data = await agent_d_format_bullet(
            original_bullet, 
            jd_data.get("domain", "SWE"), 
            jd_data.get("reqs", [])
        )
        
        # 5. Draft the Backdoor DM
        backdoor_dm = f"Hi Hiring Manager, I saw your req for {jd_data.get('domain', 'SWE')}. I just applied via your portal, but wanted to connect directly. {formatted_data.get('formatted_bullet', '')} I'd love to chat about how I can bring this to your team."
            
        return {
            "status": "READY_FOR_SUBMISSION",
            "bypasses": final_bypasses,
            "parsed_jd": jd_data,
            "ats_optimized_bullet": formatted_data.get("formatted_bullet", ""),
            "linkedin_dm_draft": backdoor_dm
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Standard library imports
import logging
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict
import re

# Third-party imports
import faiss
from huggingface_hub import InferenceClient
import numpy as np
from PyPDF2 import PdfReader
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)

# List of model names
MODELS = [
    'microsoft/phi-2',              # 0: Smallest (2.7B)
    'mistralai/Mistral-7B-Instruct-v0.2',  # 1: Medium (7B)
    'meta-llama/Llama-2-7b-chat-hf',        # 2: Also Medium (7B)
    'mistralai/Mistral-Large-Instruct-2411' # 3: Largest (11B)
]

# Select model (0 for smallest)
MODEL_INDEX = 1

# Initialize the client
client = InferenceClient(
    model=MODELS[MODEL_INDEX],
    token=os.environ["HUGGINGFACE_TOKEN"]
)

# Initialize embedding client (using a good model for embeddings)
embedding_client = InferenceClient(
    model='sentence-transformers/all-MiniLM-L6-v2',
    token=os.environ["HUGGINGFACE_TOKEN"]
)

# Initialize FAISS index
dimension = 384  # Dimension for sentence-transformers model
index = faiss.IndexFlatL2(dimension)

def generate_with_progress(prompt, max_new_tokens=100, task_name=""):
    print(f"Generating {task_name}:", end=" ", flush=True)
    response = client.text_generation(
        prompt,
        max_new_tokens=max_new_tokens,
        stream=True
    )
    
    generated_text = ""
    for token in response:
        generated_text += token
        print(".", end="", flush=True)
    print("\nDone!")
    
    return generated_text

def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = [embedding_client.feature_extraction(text) for text in texts]
    return np.array(embeddings)

def parse_cv(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error parsing CV: {str(e)}")
        raise

def extract_title_and_skills(cv_text):
    print("Starting job title and skills extraction...")
    prompt = """
    Task: From the CV text below, extract:
    1. 5 generic job titles that best match the person's experience (most recent experience is most important)
    2. A list of technical skills and programming languages (maximum 10 most important ones, based on frequency in document and recency)
    
    Format your response exactly like this:
    TITLES: [job title 1], [job title 2], [job title 3], [job title 4], [job title 5]
    SKILLS: [skill1], [skill2], [skill3], ...
    
    Note: Focus on the most recent experience when generating titles. Titles should be specific and reflect actual job roles.
    
    CV Text:
    """ + cv_text
    
    response = generate_with_progress(prompt, max_new_tokens=200, task_name="job titles and skills")
    
    # Parse the response
    lines = response.strip().split('\n')
    job_titles = []
    skills = []
    
    for line in lines:
        if "TITLES:" in line:
            job_titles = [title.strip() for title in line.replace("TITLES:", "").split(",")]
        elif "SKILLS:" in line:
            skills = [skill.strip() for skill in line.replace("SKILLS:", "").split(",")]
    
    print(f"Extracted job titles: {', '.join(job_titles)}")
    print(f"Extracted skills: {', '.join(skills)}")
    
    return job_titles, skills

def search_jobs(job_titles: List[str], skills: List[str], api_key: str) -> List[Dict]:
    """Search jobs using SerpAPI's Google Jobs API"""
    url = "https://serpapi.com/search.json"
    all_jobs = []
    
    # Combine job titles and skills into search query
    # HACK
    job_titles = ["(Senior Developer AND startup)", "(Developer AND startup)"]
    titles_query = " OR ".join(job_titles)
    skills_query = " OR ".join(skills[:3])
    search_query = f"startup AND ({titles_query}) AND ({skills_query})"
    
    params = {
        "api_key": api_key,
        "engine": "google_jobs",
        "q": search_query,
        "location": "London",
        "hl": "en",
        "gl": "uk",
        "chips": "date_posted:week"
    }
    
    try:
        while True:
            print(f"Fetching jobs page...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            jobs_list = data.get('jobs_results', [])
            print(f"Found {len(jobs_list)} jobs on this page")
            
            for job in jobs_list:
                all_jobs.append(job)
            
            # Check for next page
            if 'serpapi_pagination' in data and 'next_page_token' in data['serpapi_pagination']:
                params['next_page_token'] = data['serpapi_pagination']['next_page_token']
            else:
                break
                
            # Optional: Add delay between requests
            time.sleep(1)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching jobs: {e}")
    
    print(f"Found {len(all_jobs)} total jobs")
    return all_jobs

def compare_jobs_with_cv(jobs, cv_text, count=5):
    print("Comparing jobs with CV using LLM...", end=" ", flush=True)
    
    job_matches = []
    
    for job in jobs:
        job_description = job['description']
        
        # Create a prompt for the LLM to score the CV against the job description
        prompt = f"""
        Evaluate the suitability of the following CV for the job described below. Provide a score from 0 to 100, where 100 indicates a perfect match.
        In your response return the score only as an integer.

        CV Text:
        {cv_text}
        
        Job Description:
        {job_description}
        
        Score:
        """
        
        # Generate the score using the LLM
        response = generate_with_progress(prompt, max_new_tokens=10, task_name="scoring job match")
        
        try:
            score = int(re.findall("[0-9]{2}[0-9]?", response)[0])
        except ValueError:
            score = 0  # Default to 0 if parsing fails
        
        job_matches.append((job, score))
    
    # Sort by score descending
    job_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Take top matches
    top_matches = job_matches[:count]
    
    # Print scores for transparency
    print("\nMatch Scores:")
    for job, score in top_matches:
        print(f"{job['title']}: {score:.1f}%")
    
    return [job for job, score in top_matches]

# Main execution
if __name__ == "__main__":
    # Parse the CV
    cv_path = os.environ.get('CV_FILE_PATH', "CV.pdf")
    cv_text = parse_cv(cv_path)
    print("CV parsed successfully")

    # Extract job titles and skills
    job_titles, skills = extract_title_and_skills(cv_text)
    print(f"Generated job titles and skills: {job_titles} and {skills}")

    # Fetch jobs from APIJobs
    api_key = os.environ.get('SERP_API_KEY')
    if not api_key:
        raise ValueError("SERP_API_KEY environment variable not set")
    
    jobs = search_jobs(job_titles, skills, api_key)
    print(f"Found {len(jobs)} matching jobs")

    # Compare jobs with CV
    if not jobs:
        print("No jobs found")
        exit()
    
    top_jobs = compare_jobs_with_cv(jobs, cv_text, 10)

    # Print results
    print("\nTop Job Matches:")
    for i, job in enumerate(top_jobs, 1):
        print(f"\n{i}. Job Details:")
        for key, value in job.items():
            if len(value) < 100:  # Only print if value is short, i.e. not the full job description
                print(f"{key}: {value}")
        print("-" * 50)
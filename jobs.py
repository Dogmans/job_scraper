import logging
from huggingface_hub import InferenceClient
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events
from linkedin_jobs_scraper.filters import OnSiteOrRemoteFilters, SalaryBaseFilters, TimeFilters, RelevanceFilters
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
import faiss
import numpy as np
from PyPDF2 import PdfReader
import os

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
    1. 5 generic job titles that best match the person's experience (most recent experience is most important, job title only excluding company etc.)
    2. A list of technical skills and programming languages (maximum 10 most important ones)
    
    Format your response exactly like this:
    TITLES: [job title 1], [job title 2], [job title 3], [job title 4], [job title 5]
    SKILLS: [skill1], [skill2], [skill3], ...
    
    Note: Focus on the most recent experience when generating titles.
    
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

def add_job(data):
    url = data.link.split("?")[0]
    if url not in seen_urls:
        seen_urls.add(url)
        jobs.append({
            'title': data.title,
            'company': data.company,
            'description': data.description,
            'link': data.link
        })

def compare_jobs_with_cv(jobs, cv_text):
    cv_embedding = get_embeddings(cv_text)[0]
    job_descriptions = [job['description'] for job in jobs]
    job_embeddings = get_embeddings(job_descriptions)

    index.reset()
    index.add(job_embeddings)
    
    # Get distances and indices for top matches
    D, I = index.search(np.array([cv_embedding]), k=len(jobs))  # Get all matches
    
    # Convert distances to similarity scores (0-100)
    # FAISS returns L2 distances, so we need to convert them to similarity scores
    # Smaller distances = better match, so we invert and normalize
    max_distance = max(D[0])
    similarity_scores = [100 * (1 - (distance / max_distance)) for distance in D[0]]
    
    # Create list of (job, score) tuples
    job_matches = [(jobs[i], score) for i, score in zip(I[0], similarity_scores)]
    
    # Sort by score descending (highest score first)
    job_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 5 matches with their scores
    top_matches = job_matches[:5]
    
    # Print scores for transparency
    print("\nMatch Scores:")
    for job, score in top_matches:
        print(f"{job['title']}: {score:.1f}%")
    
    # Return jobs with their scores
    return [job for job, score in top_matches]

# Main execution
if __name__ == "__main__":
    # Parse the CV
    cv_path = os.environ.get('CV_FILE_PATH', "CV.pdf")
    cv_text = parse_cv(cv_path)
    print("CV parsed successfully")

    # Extract job title and generate similar titles
    job_titles, skills = extract_title_and_skills(cv_text)
    print(f"Generated similar job titles and skills: {job_titles} and {skills}")

    # Generate a LinkedIn query from job keywords
    linkedin_query = "(" + " OR ".join(job_titles) + ") AND (" + " OR ".join(skills) + ")"

    # Define locations and create query
    locations = ["United Kingdom"]
    query_obj = Query(
        query=linkedin_query,
        options=QueryOptions(
            locations=locations,
            # skip_promoted_jobs=True,
            filters=QueryFilters(
                relevance=RelevanceFilters.RECENT,
                time=TimeFilters.MONTH,
                base_salary=SalaryBaseFilters.SALARY_80K,
                on_site_or_remote=[OnSiteOrRemoteFilters.REMOTE, OnSiteOrRemoteFilters.HYBRID],
            ),
            limit=54
        )
    )

    # Initialize and run the scraper
    # Initialize and run the scraper with authentication
    print(os.environ['LI_AT_COOKIE'])
    jobs = []
    seen_urls = set()
    scraper = LinkedinScraper(
        # Other settings
        headless=True,
        max_workers=1,
        slow_mo=1.0,
        page_load_timeout=60
    )

    scraper.on(Events.DATA, add_job)
    scraper.on(Events.ERROR, lambda error: logging.error(f"LinkedIn scraping error: {error}"))
    scraper.on(Events.END, lambda: print("LinkedIn scraping completed"))

    print("Starting LinkedIn scraping...")
    scraper.run(query_obj)

    # Compare jobs with CV
    print("Comparing jobs with CV...")
    if not jobs:
        print("No jobs found")
        exit()
    top_jobs = compare_jobs_with_cv(jobs, cv_text)

    # Print results
    print("\nTop Job Matches:")
    for i, job in enumerate(top_jobs, 1):
        print(f"\n{i}. {job['title']} at {job['company']}")
        print(f"Link: {job['link']}")
        print("-" * 50)
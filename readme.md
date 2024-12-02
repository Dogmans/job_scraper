# Job Matching Using LLMs and SerpApi

This project aims to extract job titles from a CV, search for relevant job listings using SerpApi, and compare the CV against job descriptions using large language models (LLMs). The ultimate goal is to find the best job matches based on the candidate's experience and skills.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Functions](#functions)
- [Dependencies](#dependencies)
- [License](#license)

## Setup

1. **Clone the Repository**
    git clone https://github.com/dogmans/job_scraper.git
    cd job_scraper

2. **Run the Script**
    ./launch_jobs.ps1

## Usage

1. **Parse CV**
    Extracts the text from the given PDF CV.
    ```python
    cv_text = parse_cv(file_path)
    ```

2. **Extract Job Titles**
    Extracts job titles from the parsed CV text using a language model.
    ```python
    job_titles = extract_title(cv_text)
    ```

3. **Search Jobs**
    Searches for job listings using SerpApi based on the extracted job titles.
    ```python
    jobs = search_jobs(job_titles, api_key)
    ```

4. **Compare Jobs with CV**
    Compares the job descriptions with the CV text to find the best matches.
    ```python
    top_jobs = compare_jobs_with_cv(jobs, cv_text, count=5)
    ```

## Functions

- **parse_cv(file_path)**: Parses the CV from a PDF file.
- **extract_title(cv_text)**: Extracts job titles from the CV text.
- **search_jobs(job_titles, api_key)**: Searches for jobs using SerpApi.
- **compare_jobs_with_cv(jobs, cv_text, count)**: Compares jobs with CV to find the best matches.

## Dependencies

- Python 3.7+
- Hugging Face Hub
- numpy
- PyPDF2
- requests

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

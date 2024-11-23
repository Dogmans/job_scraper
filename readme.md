
### Running with PowerShell

Use the provided `launch_jobs.ps1` script which will prompt for your API keys and CV path.

## Features

- CV Analysis: Uses AI to extract relevant job titles and skills
- Job Search: Searches for jobs matching your profile
- Semantic Matching: Ranks jobs based on similarity to your CV
- Deduplication: Removes duplicate job listings
- Configurable: Supports various job search parameters

## Models Used

- Text Generation: Mistral 7B for CV analysis
- Embeddings: MiniLM-L6-v2 for semantic matching

## Output

The script will display:
1. Extracted job titles and skills
2. Number of jobs found
3. Top 5 matching jobs with:
   - Title
   - Company
   - Location
   - Salary (if available)
   - Posted date
   - Link
   - Required skills
   - Match score

## Troubleshooting

1. No jobs found:
   - Check your API key is valid
   - Try broadening the search criteria
   - Ensure CV is readable text (not scanned images)

2. API errors:
   - Verify environment variables are set correctly
   - Check API key permissions
   - Ensure internet connectivity

## Dependencies

- faiss-cpu
- huggingface-hub
- numpy
- PyPDF2
- requests

## License

MIT License
# Prompt for required values
$cvPath = Read-Host "Enter the path to your CV file (default is CV.pdf)"
if ([string]::IsNullOrWhiteSpace($cvPath)) { $cvPath = "CV.pdf" }

$apiKey = Read-Host "Enter your API key"
$hfToken = Read-Host "Enter your HuggingFace token"

# Set environment variables
$env:CV_FILE_PATH = $cvPath
$env:SERP_API_KEY = $apiKey
$env:HUGGINGFACE_TOKEN = $hfToken

# Check if virtual environment exists
$venvPath = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
    
    # Activate venv and install requirements
    & (Join-Path $venvPath "Scripts\Activate.ps1")
    Write-Host "Installing requirements..."
    pip install -r (Join-Path $PSScriptRoot "requirements.txt")
} else {
    # Just activate the existing venv
    & (Join-Path $venvPath "Scripts\Activate.ps1")
}

# Run the script
python "c:\Programs\tools\jobs.py"

# Keep the window open to see any output
Read-Host -Prompt "Press Enter to exit"
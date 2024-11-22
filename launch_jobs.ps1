# Function to get LinkedIn cookie from Chrome
function Get-LinkedInCookie {
    $chromePath = "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Network\Cookies"
    $edgePath = "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Network\Cookies"
    
    # Try Chrome first, then Edge
    $dbPath = if (Test-Path $chromePath) { $chromePath } else { $edgePath }
    
    # Create temp copy of cookie db (since browser might have it locked)
    $tempDB = Join-Path $env:TEMP "cookies_temp.db"
    Copy-Item -Path $dbPath -Destination $tempDB -Force
    
    try {
        # Query the cookie using SQLite
        $query = "SELECT value FROM cookies WHERE host_key LIKE '%linkedin.com' AND name='li_at'"
        $cookie = sqlite3.exe $tempDB $query
        
        if ($cookie) {
            return $cookie
        } else {
            Write-Host "LinkedIn cookie not found in browser. Please enter manually."
            return Read-Host "Enter your LinkedIn li_at cookie"
        }
    }
    catch {
        Write-Host "Error reading browser cookies. Please enter manually."
        return Read-Host "Enter your LinkedIn li_at cookie"
    }
    finally {
        # Clean up temp file
        Remove-Item $tempDB -ErrorAction SilentlyContinue
    }
}

# Prompt for required values
$cvPath = Read-Host "Enter the path to your CV file (default is CV.pdf)"
if ([string]::IsNullOrWhiteSpace($cvPath)) { $cvPath = "CV.pdf" }

# Try to get LinkedIn cookie automatically
$liAtCookie = Get-LinkedInCookie
$hfToken = Read-Host "Enter your HuggingFace token"

# Set environment variables
$env:CV_FILE_PATH = $cvPath
$env:LI_AT_COOKIE = $liAtCookie
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
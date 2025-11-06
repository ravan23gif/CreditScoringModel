# Start both frontend and backend
$env:FLASK_APP="app.py"
$env:FLASK_ENV="development"

Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory "C:\Projects\CreditScoringModel"
Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory "C:\Projects\CreditScoringModel\credit-scoring-frontend"
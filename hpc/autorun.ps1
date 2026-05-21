Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "         TS-GNN HPC AUTOMATION INIZIATA              " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " "
Write-Host "Fase 1: Configurazione Chiave SSH (Login Automatico)" -ForegroundColor Yellow
Write-Host "--> L'HPC ti chiedera la password per l'ultima volta."
Write-Host "    (Se non digiti nulla e si blocca, ricordati di inserire la tua password server)"
type "$env:USERPROFILE\.ssh\id_bocconi.pub" | ssh bocconi "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"

Write-Host " "
Write-Host "Fase 2: Zip e Sincronizzazione Codice" -ForegroundColor Yellow
.\hpc\sync_up.ps1

Write-Host " "
Write-Host "Fase 3: Setup Ambiente Conda Remoto (se necessario)" -ForegroundColor Yellow
ssh bocconi "bash ~/ts-gnn/hpc/bootstrap.sh"

Write-Host " "
Write-Host "Fase 4: Lancio del Calcolo su SLURM" -ForegroundColor Yellow
ssh bocconi "cd ~/ts-gnn && sbatch hpc/submit.sbatch"

Write-Host " "
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "OPERAZIONE COMPLETATA! Il job e' stato inviato all'HPC." -ForegroundColor Green
Write-Host "Puoi chiudere questa finestra." -ForegroundColor Green
Read-Host "Premi INVIO per uscire..."

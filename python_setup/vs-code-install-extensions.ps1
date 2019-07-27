# Script for batch installing Visual Studio Code extensions
# Specify extensions to be checked & installed by modifying $extensions

$extensions =
    # Look & Feel
    "azemoh.one-monokai",
    "pkief.material-icon-theme",
    "emmanuelbeziat.vscode-great-icons",
    "vscode-icons-team.vscode-icons",
    # Editing
    "coenraads.bracket-pair-colorizer-2",
    "remimarsal.prettier-now",
    "nobuhito.printcode",
    # Language support
    "ms-vscode.cpptools",
    "ms-vscode.powershell",
    "ms-python.python",
    # Extra functionality
    "austin.code-gnu-global",
    "grapecity.gc-excelviewer",
    "visualstudioexptteam.vscodeintellicode"

$cmd = "code --list-extensions"
Invoke-Expression $cmd -OutVariable output | Out-Null
$installed = $output -split "\s"

foreach ($ext in $extensions) {
    if ($installed.Contains($ext)) {
        Write-Host $ext "already installed." -ForegroundColor Gray
    } else {
        Write-Host "Installing" $ext "..." -ForegroundColor White
        code --install-extension $ext
    }
}
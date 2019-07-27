To set up everything for running python in vscode, do the following:

1.) Install Microsoft's Visual Studio Code (vscode):
double-click VSCodeUserSetup-x64-1.36.1.exe
Follow through with the installer, default is enough


2.) Install Code Runner extension for vscode:
Open terminal by hotkey Ctrl+` or through "View" menu
code --install-extension formulahendry.code-runner


3.) Install rest of vscode extensions using PowerShell-script:
Open "vs-code-install-extensions.ps1" in vscode
Press F1 or Ctrl+Shift+P or "View" menu to open Command Palette
Type "Run Code" and press Enter


4.) Install python using python's official 64-bit installer (version 3.7.4):
double-click "python-3.7.4-amd64.exe"
Follow through with the installer, default is good enough but could use custom install if desired


5.) Update python'installer tools:
python -m pip install --upgrade pip setuptools wheel


6.) Install all python dependencies/packages:
pip install -r requirements.txt


7.) Auto-update all installed python packages:
python update_packages.py


Done!~


PS: vscode auto-updates the extensions when you open it, no need to manually manage them


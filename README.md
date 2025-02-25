# Road Defect Indexing System

## Cloning the Repository
To properly clone this repository along with its submodules, run:
```sh
git clone --recurse-submodules https://github.com/ColasRab/Road-Defect-Indexing-System.git
```

### If You Cloned Without Submodules
If you forgot to include submodules when cloning, run:
```sh
git submodule update --init --recursive
```

### Updating Submodules
To update the submodules to the latest version, run:
```sh
git submodule update --remote --merge
```

## Setting Up the Virtual Environment
After cloning the repository, run the setup script to automatically create and activate the virtual environment:

### **For Windows**
```sh
setup.bat
```
To manually activate the virtual environment after running the script:
```sh
venv\Scripts\activate
```

### **For Linux/macOS**
```sh
bash setup.sh
```
To manually activate the virtual environment after running the script:
```sh
source venv/bin/activate
```

## Installing Dependencies
The setup script will also install all required dependencies automatically. However, if you need to manually install them, run:
```sh
pip install -r requirements.txt
```

## Running the Project
After setting up, you can start using the project. Ensure that the virtual environment is activated before running any scripts.

To deactivate the virtual environment, use:
```sh
deactivate
```

---
Now your environment is set up and ready to go!


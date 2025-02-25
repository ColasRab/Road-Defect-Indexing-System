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
After cloning the repository, create and activate a virtual environment:

### **For Windows**
```sh
python -m venv venv
venv\Scripts\activate
```

### **For Linux/macOS**
```sh
python3 -m venv venv
source venv/bin/activate
```

## Installing Dependencies
Once the virtual environment is activated, install all required dependencies:
```sh
pip install -r requirements.txt
```

## Running the Project
After installing dependencies, you can start using the project. Ensure that the virtual environment is activated before running any scripts.

To deactivate the virtual environment, use:
```sh
deactivate
```

---
Now your environment is set up and ready to go!


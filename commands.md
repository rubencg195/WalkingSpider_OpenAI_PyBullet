
# To use this environment
```python
conda env create -f <environment-name>.yml

pip install -r requirements.txt
```

---

# Commands

### Clone
```python
conda create --name new_name --clone old_name
```

### Create a conda environment
```python
conda create --name <environment-name> python=<version:2.7/3.5>
```

### To create a requirements.txt file:
```python
conda list #Gives you list of packages used for the environment

conda list -e > requirements.txt #Save all the info about packages to your folder
```

### To export environment file
```python
activate <environment-name>
conda env export > <environment-name>.yml
```
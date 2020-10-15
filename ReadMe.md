# Walking Spider

---

## See the original [ReadMe.md](./ReadMe_orig.md)

---

## Steps taken to run `test_gym_spider_env.py`
- Run  
    ```python
    $ conda env create -f <environment-name>.yml
    $ pip install -r requirements.txt
    ```
- Run `pip install .` inside `walking-spider/`
- while running the example tensorflow was degraded to 1.14, due to tensorflow.contrib error  
    ```python
    $ pip uninstall tensorflow
    $ pip install tensorflow==1.14
    ```
- **IMPORTANT -** local import (`import walking_spider`) should work now, for importing it from anywhere copy `walking_spider` to your conda env
	- For example in linux, copy destination would be  
	  `/home/<user>/anaconda3/envs/<envName>/python<version>/` 

### Running the Script
- before running make sure `import walking_spider` works outside local dir
- install any pending requirements from requirements.txt, or while running the example
- Run `test_gym_spider_env.py` for a pre-trained spider

---

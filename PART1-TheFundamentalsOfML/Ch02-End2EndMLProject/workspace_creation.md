<h1 align="center">Creating the Workspace</h1>

This markdown contains two sections:
1. [Creating Workspace](#Creating-Workspace) - If you want all your libraries of ML to be accessible everywhere
2. [Creating an Isolated Workspace](Creating-an-Isolated-Workspace) - If you want an isolated environment for each project or for similar type of projects. This is **Strongly Recommended**.

Before starting the main Workspace Creation:

First you will need to have Python installed. If not installed, download and install from [here](https://www.python.org/). Latest version of Python 3 is recommended!

# Creating Workspace
* Make a workspace directory for Machine Learning Projects and Codes:

```
$ export ML_PATH="$HOME/ml"     # you can change the path if you prefer
$ mkdir -p $ML_PATH
```

* We will be needing a bunch of modules (wheel files) like - Scikit-Learn, Numpy, Pandas, Jupyter and Matplotlib. You can install them using Anaconda or Python's own packaging system *pip*. **For windows please see the note at the end of this section.**

```
$ pip3 --version    # checking for pip
pip 9.0.1 from [...]/lib/python3.5/site-packages (python 3.5)
$ pip3 install --upgrade pip
Collecting pip
[...]
Successfully installed pip-10.0.1
$ pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
[...]
```

* If pip is not available
```
$ cd ~
$ wget https://bootstrap.pypa.io/get-pip.py
$ python3 get-pip.py
$ pip3 install --upgrade pip
[...]
Collecting pip
[...]
Successfully installed pip-10.0.1
$ pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
[...]
```


* Checking the installed packages

```
$ python3 -c "import jupyter, matplotlib, numpy, pandas, scipy, scikit-learn"
```


* Starting up the jupyter notebook

```
$ cd $ML_PATH
$ jupyter notebook
[...]
```


* A jupyter server is now running in the terminal. Go to http://localhost:8888/ (Usually happens automatically).

***
> Side note: On windows some wheels might cause error at time of installation. You can install pre-compiled wheel files for those by downloading them from [this](https://www.lfd.uci.edu/~gohlke/pythonlibs/) website.

Example: If you want to download numpy+mkl on your windows system go to the [link](https://www.lfd.uci.edu/~gohlke/pythonlibs/) and find numpy+mkl on the site. There will be some countable number of numpy+mkl files but you have to select one that meets your specification. If you have python 3.5 64 bit version you will download `numpy‑1.14.5+mkl‑cp35‑cp35m‑win_amd64.whl`, if you have python 2.7 32 bit you will download `numpy‑1.14.5+mkl‑cp27‑cp27m‑win32.whl`
***

## Creating an Isolated Workspace
* Make a workspace directory for Machine Learning Projects and Codes:

```
$ export ML_PATH="$HOME/ml"     # you can change the path if you prefer
$ mkdir -p $ML_PATH
```

* We will be needing a bunch of modules (wheel files) like - Scikit-Learn, Numpy, Pandas, Jupyter and Matplotlib. You can install them using Anaconda or Python's own packaging system *pip*.

```
$ pip3 --version    # checking for pip
pip 9.0.1 from [...]/lib/python3.5/site-packages (python 3.5)
$ pip3 install --upgrade pip
Collecting pip
[...]
Successfully installed pip-10.0.1
$ pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
[...]
```

* If pip is not available

```
$ cd ~
$ wget https://bootstrap.pypa.io/get-pip.py
$ python3 get-pip.py
$ pip3 install --upgrade pip
[...]
Collecting pip
[...]
Successfully installed pip-10.0.1
$ pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
[...]
```

* Installing virtualenv (You might need admin rights here)

```
$ pip3 install virtualenv virtualenvwrapper
```

* Now create an isolated environment

```
$ cd $ML_PATH
$ virtual env
Using base prefix '[...]'
New python executable in [...]/ml/env/bin/python3.5
Also creating executable in [...]/ml/env/bin/python3.5
Installing setuptools, pip, wheel...done.
```

* Now everytime you want to activate the environment

```
$ cd $ML_PATH
$ source env/bin/activate
```
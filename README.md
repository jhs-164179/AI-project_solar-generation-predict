### Requirements

| Software      	| Version 	|
|---------------	|---------	|
| Python        	| 3.9.13  	|
| Tensorflow-gpu       	| 2.10.1  	|
| Keras 	| 2.10.0   	|
| Scikit-Learn        	| 1.1.2  	|
| PyMySQL        	| 1.1.1  	|
| MYSQL        	| 8.0.32  	|

### Dataset

- `https://www.data.go.kr/data/15099650/fileData.do` Solar power generation data.
- `https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36` Weather data.

### Installation

  ```sh
  pip install -r requirements.txt
  ```

### Structures

- `Flask-based-AI-server` AI server.
- `Flask-based-AI-server/main.py` Code for run AI server.
- `Flask-based-AI-server/model.py` Proposed Hybrid model code.
- `Flask-based-AI-server/module.py` NLinear, DLinear implementation by tensorflow.
- `Flask-based-AI-server/utils.py` Code for fix random seed and normalization.

- `Flask-based-Web-Application-server` Web Application Server (WAS).
- `Flask-based-Web-Application-server/app.py` Code for run WAS.

### Getting Started

Plz check the IP information before run.
IP information is in here.

- `Flask-based-AI-server/main.py`
- `Flask-based-Web-Application-server/app.py`
- `Flask-based-Web-Application-server/template/index.html`

For testing with dataset, upload data by using button in html after run WAS

### Run

  ```sh
  # for run AI server
  python Flask-based-AI-server/main.py

  # for run WAS
  python Flask-based-Web-Application-server/app.py

  # access
  # localhost/demo
  ```
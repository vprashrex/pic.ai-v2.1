import sys
import multiprocessing
import uvicorn
from src import app

BASE_DIR = "./src/"
PORT = 5000

sys.path.append(BASE_DIR)

bind = "0.0.0.0:{}".format(PORT)
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'gevent'
worker_connections = 1000
timeout = 10000

uvicorn.run(app,port=5000)
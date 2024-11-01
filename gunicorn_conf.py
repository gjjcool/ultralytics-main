import os
import multiprocessing

daemon = True  # 设置守护进程
bind = '127.0.0.1:8000'  # 监听内网端口8000
chdir = './'  # 工作目录
worker_class = 'uvicorn.workers.UvicornWorker'  # 工作模式
# workers = multiprocessing.cpu_count() + 1  # 并行工作进程数 核心数*2+1个
workers = 4
threads = 2  # 指定每个工作者的线程数
worker_connections = 2000  # 设置最大并发量
loglevel = 'debug'  # 错误日志的日志级别
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
# 设置访问日志和错误信息日志路径
log_dir = "./log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
pidfile = './log/gunicorn.pid'  # 设置进程文件目录
accesslog = "./log/gunicorn_access.log"
errorlog = "./log/gunicorn_error.log"

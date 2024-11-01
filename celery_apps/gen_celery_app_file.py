# generate.py
from jinja2 import Environment, FileSystemLoader
import os
import json

# 设置模板文件所在的目录
template_dir = os.path.dirname(os.path.abspath(__file__))

# 创建 Jinja2 环境
env = Environment(loader=FileSystemLoader(template_dir))

# 加载模板
template = env.get_template('template.py.j2')
bg_task_template = env.get_template('bg_task_template.py.j2')

with open('../zysd_server_conf.json', 'r', encoding='utf-8') as conf_file:
    conf = json.load(conf_file)

for i in range(conf['celery_instance_num']):
    output = template.render(i=i)
    with open(f'celery_app_{i}.py', 'w', encoding='utf-8') as f:
        f.write(output)

with open(f'../bg_task.py', 'w', encoding='utf-8') as f:
    f.write(bg_task_template.render(inds=range(conf['celery_instance_num'])))



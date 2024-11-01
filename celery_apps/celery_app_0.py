if __name__ == '__main__':
    import eventlet

    eventlet.monkey_patch()
    from predict_algo import predict_page

import os
import sys
import traceback

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from celery import Celery
from UltralyticsYOLO.toolkit.BgTask import BgTask
import redis_lock
from celery_app_recall import recall

celery_app_0 = Celery('celery_app_0', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')


@celery_app_0.task(name='celery_apps.celery_app_0.predict_bt')
def predict_bt(filename, pageId, bucketName, graphId, host):
    pageData = dict()
    try:
        pageData = predict_page(filename, pageId, bucketName, host)
    except Exception as e:
        pageData['graphPageId'] = pageId
        pageData['isSuccess'] = False
        pageData['errorInfo'] = traceback.format_exc()
        print(pageData['errorInfo'])
    finally:
        with redis_lock.Lock(redis_client, graphId):
            print(f'[{graphId}] with lock...')
            bgTask = load_task_state(graphId)
            if bgTask:
                if bgTask.appendPage(pageData):
                    redis_client.delete(bgTask.graphId)
                    recall.delay(bgTask.data, bgTask.callbackUrl)
                else:
                    save_task_state(bgTask)
            else:
                print('Error::Load BgTask Fail.')


def save_task_state(bg_task: BgTask):
    serialized_task = json.dumps(bg_task.__dict__)
    redis_client.set(bg_task.graphId, serialized_task)


def load_task_state(task_id):
    serialized_task = redis_client.get(task_id)
    if serialized_task:
        data = json.loads(serialized_task)
        bg_task = BgTask(**data)
        return bg_task
    return None


if __name__ == '__main__':
    import json
    import os
    import threading
    import redis
    with open('zysd_server_conf.json', 'r', encoding='utf-8') as conf_file:
        conf = json.load(conf_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    redis_client = redis.StrictRedis(host='localhost', port=6379, db=15)

    app_name = os.path.splitext(os.path.basename(__file__))[0]

    def start_worker(i):
        print(f'--hostname={app_name}_worker{i + 1}@%h')
        celery_app_0.worker_main(
            argv=['worker', '--loglevel=info', f'--concurrency={conf["celery_concurrency"]}',
                  f'--pool={conf["celery_pool"]}', f'--hostname={app_name}_worker{i + 1}@%h'])


    threads = []
    for i in range(conf['celery_workers']):
        thread = threading.Thread(target=start_worker, args=(i,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
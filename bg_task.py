
import celery_apps.celery_app_0 as app0

import celery_apps.celery_app_1 as app1

import celery_apps.celery_app_2 as app2

import celery_apps.celery_app_3 as app3

import celery_apps.celery_app_4 as app4

import celery_apps.celery_app_5 as app5

import celery_apps.celery_app_6 as app6


from UltralyticsYOLO.toolkit.BgTask import BgTask

import redis
import json
import redis_lock

from celery_apps.predict_algo import download_file

pageIdSeparator = ','


apps = [

    app0,

    app1,

    app2,

    app3,

    app4,

    app5,

    app6,

]
app_num = len(apps)
call_id = 0

redis_client = redis.StrictRedis(host='localhost', port=6379, db=15)


def save_task_state(bg_task: BgTask):
    print(f'[{bg_task.graphId}] Init BgTask')
    serialized_task = json.dumps(bg_task.__dict__)
    with redis_lock.Lock(redis_client, bg_task.graphId):
        redis_client.set(bg_task.graphId, serialized_task)


def bg_task(files, pageIdList, graphId, callbackUrl, bucketName, host):
    global call_id
    fileNameList = files.split(pageIdSeparator)
    pageIdList = pageIdList.split(pageIdSeparator)
    bgTask = BgTask(graphId, len(fileNameList), callbackUrl)
    save_task_state(bgTask)
    for filename, pageId in zip(fileNameList, pageIdList):
      apps[call_id].predict_bt.delay(filename, pageId, bucketName, graphId, host)
      call_id = (call_id + 1) % app_num

import os

from celery import Celery
import httpx
import httpcore
import time
import random

celery_app_recall = Celery('celery_app_recall',
                           broker='redis://localhost:6379/13',
                           backend='redis://localhost:6379/13')

callbackPatience = 10


@celery_app_recall.task
def recall(data, url):
    callbackCnt = 0
    with httpx.Client() as client:
        while callbackCnt < callbackPatience:
            callbackCnt += 1
            try:
                start_time = time.time()
                response = client.post(url, json=data, timeout=10 + 20 * callbackCnt)

            except (httpx.ConnectTimeout, httpcore.ConnectTimeout, httpx.ReadError, httpcore.ReadError,
                    httpx.ReadTimeout, httpcore.ReadTimeout, httpx.WriteError, httpcore.WriteError,
                    httpx.WriteTimeout, httpcore.WriteTimeout, httpx.RemoteProtocolError,
                    httpcore.RemoteProtocolError):
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(
                    f"[duration: {elapsed_time:.2f}] [{data['graphId']}] Retrying {callbackCnt}/{callbackPatience}...")
                if callbackCnt >= callbackPatience:
                    print(f"[duration: {elapsed_time:.2f}] [{data['graphId']}] Exceed Max Retrying Patience")
                    raise
            except Exception as e:
                print(f"Httpx other error occurred: {type(e).__name__}")
                raise
            else:
                if response.status_code == 200:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"[duration: {elapsed_time:.2f}] [{data['graphId']}] CALLBACK_SUCCESS____")
                    return
            print('RE_CALLBACK____')
            time.sleep(random.randint(callbackCnt, 15))


if __name__ == '__main__':
    celery_app_recall.worker_main(
        argv=['worker', '--loglevel=info', f'--concurrency={os.cpu_count()}',
              f'--pool=prefork', f'--hostname=celery_app_recall_worker{0}@%h']
    )

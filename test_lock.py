import redis
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('id', type=str)
args = parser.parse_args()
id = args.id

redis_client = redis.StrictRedis(host='localhost', port=6379, db=15)

lock = redis_client.lock(id, blocking=True)
if lock.acquire(blocking=True):
    print(f'[{id}] YES')
    lock.release()
else:
    print('NO')

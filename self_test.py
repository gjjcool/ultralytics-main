import asyncio
import json
import httpx
import UltralyticsYOLO.toolkit.comm_toolkit as comm

with open('zysd_server_conf.json', 'r', encoding='utf-8') as conf_file:
    conf = json.load(conf_file)


async def self_test_predict_interface():
    print('self_test_predict_interface')
    url = f"http://{conf['host']}:{conf['port']}/predict"
    data = {
        "files": "/202408/40c57edb788a4b649d68bfcdca0fbbcc.png",  # 用英文逗号分隔的文件名称字符串
        "pageIdList": "TEST",  # 用英文逗号分隔的页面ID字符串
        "graphId": comm.get_time_str(),  # 图纸请求编号
        "callbackUrl": f"http://{conf['host']}:{conf['port']}/callback",  # 回调URL
        "bucketName": "cisdi-drawing"  # MinIO 桶名
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=data)
        print(f"Status code: {response.status_code}")


asyncio.run(self_test_predict_interface())

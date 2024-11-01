import httpx
import httpcore
import time
import random
import sys

sys.path.append('UltralyticsYOLO/toolkit')
from predict import ResultFileGen

callbackPatience = 10


class BgTask:
    def __init__(self, graphId, fileNum, callbackUrl, data=None):
        self.graphId = graphId

        # self.data_dict = ResultFileGen(graphId)
        if data is None:
            self.data = dict()
            self.data['graphId'] = graphId
            self.data['pageData'] = []
            self.data['isSuccess'] = True
            self.data['errorInfo'] = ''
        else:
            self.data = data

        self.fileNum = fileNum
        self.callbackUrl = callbackUrl

    def appendPage(self, pageData):
        # self.data_dict.appendPage(pageData)
        self.data['pageData'].append(pageData)

        return len(self.data['pageData']) == self.fileNum

    # def recall(self):
    #     callbackCnt = 0
    #     with httpx.Client() as client:
    #         while callbackCnt < callbackPatience:
    #             callbackCnt += 1
    #             try:
    #                 start_time = time.time()
    #                 response = client.post(self.callbackUrl, json=self.data, timeout=10 + 20 * callbackCnt)
    #
    #             except (httpx.ConnectTimeout, httpcore.ConnectTimeout, httpx.ReadError, httpcore.ReadError,
    #                     httpx.ReadTimeout, httpcore.ReadTimeout, httpx.WriteError, httpcore.WriteError,
    #                     httpx.WriteTimeout, httpcore.WriteTimeout, httpx.RemoteProtocolError,
    #                     httpcore.RemoteProtocolError):
    #                 end_time = time.time()
    #                 elapsed_time = end_time - start_time
    #                 print(
    #                     f"[duration: {elapsed_time:.2f}] [{self.graphId}] Retrying {callbackCnt}/{callbackPatience}...")
    #                 if callbackCnt >= callbackPatience:
    #                     print(f"[duration: {elapsed_time:.2f}] [{self.graphId}] Exceed Max Retrying Patience")
    #                     raise
    #             except Exception as e:
    #                 print(f"Httpx other error occurred: {type(e).__name__}")
    #                 raise
    #             else:
    #                 if response.status_code == 200:
    #                     end_time = time.time()
    #                     elapsed_time = end_time - start_time
    #                     print(f'[duration: {elapsed_time:.2f}] [{self.graphId}] CALLBACK_SUCCESS____')
    #                     return
    #             print('RE_CALLBACK____')
    #             time.sleep(random.randint(callbackCnt, 15))

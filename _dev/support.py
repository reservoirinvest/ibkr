from ib_insync import IB, util
import time
# import asyncio

import json

with open('var.json', 'r') as fp:
    data = json.load(fp)

market = 'snp'

host = data['common']['host']
port = data[market]['port']
cid = 0

ib = IB().connect(host=host, port=port, clientId=cid)
print(ib.isConnected())
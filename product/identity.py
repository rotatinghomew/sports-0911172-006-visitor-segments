from .store import require, money

class Identity:
    def __init__(self, store):
        self.store = store

    def connect(self, id, devices, consent=True):
        require(bool(devices) and len(devices)==len(set(devices)),'设备集合无效')
        return self.store.put('identity',id,dict(devices=devices,consent=bool(consent)),create=True)

    def split(self, id, devices):
        record=self.store.get('identity',id)
        require(set(devices)<=set(record['devices']),'不能添加未关联设备')
        record['devices']=devices
        return self.store.put('identity',id,record)

    def withdraw(self, id):
        record=self.store.get('identity',id)
        record['consent']=False
        return self.store.put('identity',id,record)

    def owners(self, device):
        return [record['id'] for record in self.store.all('identity')
                if record['consent'] and device in record['devices']]

from .store import require, money

class Audiences:
    def __init__(self, store):
        self.store=store

    def save(self, id, name, members, reason):
        require(bool(name) and bool(reason),'名单说明为空')
        require(len(members)==len(set(members)),'名单重复')
        for member in members:
            require(self.store.get('identity',member)['consent'],'成员无分析许可')
        return self.store.put('audience',id,dict(name=name,members=members,reason=reason),create=True)

    def replace(self, id, members, reason):
        require(len(members)==len(set(members)) and bool(reason),'替换说明或名单无效')
        for member in members:
            require(self.store.get('identity',member)['consent'],'成员无分析许可')
        record=self.store.get('audience',id)
        record.update(members=members,reason=reason)
        return self.store.put('audience',id,record)

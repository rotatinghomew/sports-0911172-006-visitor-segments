from .store import require, money

class Outreach:
    def __init__(self, store):
        self.store=store

    def queue(self, id, audience, content):
        record=self.store.get('audience',audience)
        require(bool(content),'消息为空')
        return self.store.put('mailing',id,dict(audience=audience,members=record['members'],
            content=content,state='queued'),create=True)

    def send(self, id):
        with self.store.atomic():
            record=self.store.get('mailing',id)
            if record['state']=='sent':
                return record
            sent=[]
            for member in record['members']:
                if self.store.get('identity',member)['consent']:
                    self.store.put('delivery',f'{id}:{member}',dict(mailing=id,member=member),create=True)
                    sent.append(member)
            record.update(state='sent',sent=sent)
            return self.store.put('mailing',id,record)

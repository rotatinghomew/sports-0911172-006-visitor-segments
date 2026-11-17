from .store import require, money

class Visits:
    def __init__(self, store):
        self.store=store

    def record(self, id, device, venue, at, activity, cents=0):
        require(activity in ('match','exercise','mall','other'),'活动类型无效')
        return self.store.put('visit',id,dict(device=device,venue=venue,at=at,
            activity=activity,cents=money(cents)),create=True)

    def venue_totals(self):
        totals={}
        for record in self.store.all('visit'):
            totals[record['venue']]=totals.get(record['venue'],0)+1
        return totals

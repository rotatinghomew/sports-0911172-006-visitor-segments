import json
import unittest
from pathlib import Path
from product.app import Application


class BusinessBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.app=Application()
        self.app.replay(json.loads((Path(__file__).parent.parent/'data'/'demo.json').read_text()))

    def tearDown(self):
        self.app.store.close()
    def test_shared_device_has_multiple_candidate_owners(self):
        self.app.execute('identity.connect',dict(id='p3',devices=['d1']))
        self.assertEqual(self.app.execute('identity.owners',dict(device='d1')),['p2','p3'])

    def test_send_retries_do_not_duplicate_messages(self):
        before=self.app.store.snapshot()
        self.app.execute('outreach.send',dict(id='m1'))
        self.assertEqual(self.app.store.snapshot(),before)

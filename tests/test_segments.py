import json
import unittest
from pathlib import Path
from product.app import Application

WEEKS = ['2026-07-01', '2026-07-06', '2026-07-13', '2026-07-20']


def connect(app, id, devices, consent=True):
    return app.execute('identity.connect', dict(id=id, devices=devices, consent=consent))


def visit(app, id, device, activity, at, cents=0, venue='v1'):
    return app.execute('visits.record', dict(id=id, device=device, venue=venue, at=at,
                                             activity=activity, cents=cents))


def build_groups(app, sizes=(6, 6, 6), activities=('match', 'exercise', 'mall'),
                 weeks=WEEKS[:2], prefix='g'):
    """构造若干行为纯粹的群体：每人独立设备、组内活动一致。返回 {组号: [主体编号]}。"""
    groups = {}
    for g, (size, activity) in enumerate(zip(sizes, activities)):
        members = []
        for i in range(size):
            member = f'{prefix}{g}_{i}'
            connect(app, member, [f'd_{member}'])
            for v, week in enumerate(weeks):
                visit(app, f'v_{member}_{v}', f'd_{member}', activity, week)
            members.append(member)
        groups[g] = members
    return groups


class SegmentsTestCase(unittest.TestCase):
    def tearDown(self):
        self.app.store.close()


class AttributionTests(SegmentsTestCase):
    def setUp(self):
        self.app = Application()
        connect(self.app, 'p1', ['d1', 'fam'])
        connect(self.app, 'p2', ['d2', 'fam'], consent=False)
        connect(self.app, 'p3', ['d3'])
        connect(self.app, 'p4', ['d4'])
        connect(self.app, 'p5', ['d5'], consent=False)
        for i in range(3):
            visit(self.app, f'v1_{i}', 'd1', 'match', WEEKS[0])
        for i in range(2):
            visit(self.app, f'vf_{i}', 'fam', 'mall', WEEKS[0])
        visit(self.app, 'v2', 'd2', 'exercise', WEEKS[0])
        visit(self.app, 'vx', 'stray', 'other', WEEKS[0])
        for i in range(4):
            visit(self.app, f'v3_{i}', 'd3', 'mall', WEEKS[0])

    def test_shared_and_unlicensed_visits_are_not_attributed(self):
        record = self.app.execute('segments.analyze', dict(id='s1', k=2))
        coverage = record['coverage']
        self.assertEqual(coverage['identities_total'], 5)
        self.assertEqual(coverage['consenting'], 3)
        self.assertEqual(coverage['excluded_no_consent'], 2)
        self.assertEqual(coverage['assigned'], 2)
        self.assertEqual(coverage['unassigned_low_data'], 1)
        self.assertEqual(coverage['visits_total'], 11)
        self.assertEqual(coverage['visits_attributed'], 7)
        self.assertEqual(coverage['visits_ambiguous_device'], 2)
        self.assertEqual(coverage['visits_unknown_device'], 1)
        self.assertEqual(coverage['visits_excluded_no_consent'], 1)
        members = {m for c in record['clusters'] for m in c['members']}
        self.assertEqual(members, {'p1', 'p3'})
        self.assertEqual(record['unassigned'], ['p4'])
        dumped = json.dumps(record)
        for excluded in ('p2', 'p5'):
            self.assertNotIn(excluded, dumped)

    def test_shared_device_only_members_cannot_be_clustered(self):
        app = Application()
        try:
            connect(app, 'q1', ['fam'])
            connect(app, 'q2', ['fam'])
            for i in range(4):
                visit(app, f'vq_{i}', 'fam', 'match', WEEKS[0])
            before = app.store.snapshot()
            with self.assertRaises(ValueError):
                app.execute('segments.analyze', dict(id='s2', k=2))
            self.assertEqual(app.store.snapshot(), before)
        finally:
            app.store.close()

    def test_analyze_is_deterministic(self):
        first = self.app.execute('segments.analyze', dict(id='s1', k=2))
        other = Application()
        try:
            connect(other, 'p1', ['d1', 'fam'])
            connect(other, 'p2', ['d2', 'fam'], consent=False)
            connect(other, 'p3', ['d3'])
            connect(other, 'p4', ['d4'])
            connect(other, 'p5', ['d5'], consent=False)
            for i in range(3):
                visit(other, f'v1_{i}', 'd1', 'match', WEEKS[0])
            for i in range(2):
                visit(other, f'vf_{i}', 'fam', 'mall', WEEKS[0])
            visit(other, 'v2', 'd2', 'exercise', WEEKS[0])
            visit(other, 'vx', 'stray', 'other', WEEKS[0])
            for i in range(4):
                visit(other, f'v3_{i}', 'd3', 'mall', WEEKS[0])
            second = other.execute('segments.analyze', dict(id='s1', k=2))
        finally:
            other.store.close()
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))


class ParameterTests(SegmentsTestCase):
    def setUp(self):
        self.app = Application()
        build_groups(self.app)

    def test_invalid_parameters_are_rejected_without_mutation(self):
        before = self.app.store.snapshot()
        for arguments in (dict(id='b1', k=1), dict(id='b2', k=0), dict(id='b3', k=2.5),
                          dict(id='b4', k=2, features=['share_match', 'bogus']),
                          dict(id='b5', k=2, features=[]),
                          dict(id='b6', k=2, min_visits=0),
                          dict(id='b7', k=2, ks=[2, 99])):
            with self.assertRaises(ValueError, msg=arguments):
                self.app.execute('segments.analyze', arguments)
        self.assertEqual(self.app.store.snapshot(), before)

    def test_duplicate_segmentation_id_is_rejected(self):
        self.app.execute('segments.analyze', dict(id='s1', k=2))
        before = self.app.store.snapshot()
        with self.assertRaises(ValueError):
            self.app.execute('segments.analyze', dict(id='s1', k=2))
        self.assertEqual(self.app.store.snapshot(), before)

    def test_feature_subset_is_recorded_and_comparable(self):
        full = self.app.execute('segments.analyze', dict(id='full', k=3))
        subset = self.app.execute('segments.analyze', dict(
            id='subset', k=3, features=['share_match', 'share_exercise', 'share_mall', 'share_other']))
        self.assertEqual(subset['features'], ['share_match', 'share_exercise', 'share_mall', 'share_other'])
        comparison = self.app.execute('segments.correspond', dict(before='full', after='subset'))
        self.assertTrue(all(c['kind'] == 'preserved' for c in comparison['clusters']))
        explain = self.app.execute('segments.explain', dict(id='subset'))
        self.assertEqual(sorted(explain['feature_labels']),
                         ['share_exercise', 'share_mall', 'share_match', 'share_other'])
        self.assertEqual(len(full['candidates']), 3)
        for candidate in full['candidates']:
            self.assertIn('silhouette', candidate)
            self.assertIn('stability', candidate)


class CorrespondTests(SegmentsTestCase):
    def setUp(self):
        self.app = Application()
        self.groups = build_groups(self.app)

    def kinds(self, result):
        return [c['kind'] for c in result['clusters']]

    def test_split_and_merge_are_reported_not_renumbered(self):
        self.app.execute('segments.analyze', dict(id='s1', k=2))
        self.app.execute('segments.analyze', dict(id='s2', k=3))
        split_view = self.app.execute('segments.correspond', dict(before='s1', after='s2'))
        self.assertEqual(sorted(self.kinds(split_view)), ['preserved', 'split'])
        self.assertFalse(split_view['stable'])
        self.assertTrue(any('不能沿用旧群号' in a for a in split_view['alerts']))
        merge_view = self.app.execute('segments.correspond', dict(before='s2', after='s1'))
        self.assertEqual(sorted(self.kinds(merge_view)), ['merged', 'merged', 'preserved'])
        self.assertEqual(len(merge_view['matrix']), 3)
        self.assertEqual(len(merge_view['matrix'][0]), 2)

    def test_lost_correspondence_after_withdrawal(self):
        record = self.app.execute('segments.analyze', dict(id='s1', k=3))
        withdrawn = record['clusters'][1]['members']
        for member in withdrawn:
            self.app.execute('identity.withdraw', dict(id=member))
        self.app.execute('segments.analyze', dict(id='s2', k=3))
        result = self.app.execute('segments.correspond', dict(before='s1', after='s2'))
        kinds = {c['cluster']: c['kind'] for c in result['clusters']}
        self.assertEqual(kinds[1], 'lost')
        self.assertEqual(kinds[0], 'preserved')
        self.assertEqual(kinds[2], 'preserved')
        self.assertTrue(any('丢失' in a for a in result['alerts']))

    def test_identical_segmentation_is_stable(self):
        self.app.execute('segments.analyze', dict(id='s1', k=3))
        result = self.app.execute('segments.correspond', dict(before='s1', after='s1'))
        self.assertTrue(result['stable'])
        self.assertEqual(self.kinds(result), ['preserved'] * 3)


class StabilityAndNewcomerTests(SegmentsTestCase):
    def setUp(self):
        self.app = Application()
        self.groups = build_groups(self.app, sizes=(4, 4, 4), weeks=WEEKS)
        for i in range(2):
            connect(self.app, f'n{i}', [f'dn{i}'])
            for v, week in enumerate(WEEKS[2:]):
                visit(self.app, f'vn{i}_{v}', f'dn{i}', 'exercise', week)
                visit(self.app, f'vn{i}b_{v}', f'dn{i}', 'exercise', week)
        connect(self.app, 'n2', ['dn2'])
        visit(self.app, 'vn2', 'dn2', 'mall', WEEKS[3])

    def test_cross_week_stability_and_newcomer_coverage(self):
        record = self.app.execute('segments.analyze', dict(id='s1', k=3, min_visits=2))
        self.assertEqual(record['stability']['agreement'], 1.0)
        self.assertEqual(record['stability']['weeks_train'], 2)
        self.assertEqual(record['stability']['weeks_total'], 4)
        self.assertEqual(record['newcomers'], {'count': 3, 'assigned': 2, 'unassigned': 1})
        self.assertEqual(record['unassigned'], ['n2'])
        sizes = sorted(c['size'] for c in record['clusters'])
        self.assertEqual(sizes, [4, 4, 6])

    def test_single_week_data_reports_no_stability(self):
        app = Application()
        try:
            build_groups(app, sizes=(3, 3), activities=('match', 'mall'), weeks=WEEKS[:1], prefix='w')
            record = app.execute('segments.analyze', dict(id='sw', k=2))
            self.assertIsNone(record['stability'])
            self.assertIsNone(record['newcomers'])
        finally:
            app.store.close()


class ExplainTests(SegmentsTestCase):
    def setUp(self):
        self.app = Application()
        build_groups(self.app, sizes=(4, 4), activities=('match', 'exercise'), prefix='e')
        for i in range(3):
            member = f'mall_{i}'
            connect(self.app, member, [f'd_{member}'])
            cents = 500 if i == 0 else 0
            visit(self.app, f'vm_{i}_0', f'd_{member}', 'mall', WEEKS[0], cents=cents)
            visit(self.app, f'vm_{i}_1', f'd_{member}', 'mall', WEEKS[1])
        self.record = self.app.execute('segments.analyze', dict(id='s1', k=3))
        self.explain = self.app.execute('segments.explain', dict(id='s1'))

    def test_explain_contains_only_aggregates(self):
        dumped = json.dumps(self.explain, ensure_ascii=False)
        self.assertNotIn('"members"', dumped)
        self.assertNotIn('"device"', dumped)
        for cluster in self.explain['clusters']:
            self.assertEqual(set(cluster['profile']), set(self.record['clusters'][0]['profile']))
            for contrast in cluster['contrasts']:
                self.assertIn(contrast['direction'], ('偏高', '偏低'))
                self.assertIn('label', contrast)

    def test_small_cluster_single_purchase_is_flagged(self):
        mall_cluster = next(c for c in self.explain['clusters']
                            if c['uncertainty']['purchase_visits'] == 1)
        self.assertEqual(mall_cluster['size'], 3)
        self.assertTrue(any('单次' in w or '一次消费' in w for w in mall_cluster['warnings']))
        self.assertTrue(any('样本量较小' in w for w in mall_cluster['warnings']))
        self.assertTrue(any('低数据' in w for w in mall_cluster['warnings']))
        summary = next(c for c in self.explain['clusters']
                       if c['uncertainty']['purchase_visits'] == 1)['summary']
        self.assertIn('商圈类到访占比', summary)

    def test_coverage_and_candidates_are_exposed(self):
        coverage = self.explain['coverage']
        self.assertEqual(coverage['assigned'], 11)
        self.assertEqual(len(self.explain['candidates']), 3)
        self.assertEqual(self.explain['stability']['agreement'], 1.0)


class MigrationTests(SegmentsTestCase):
    def setUp(self):
        self.app = Application()
        for i, activity in enumerate(('match', 'match', 'mall', 'mall')):
            member = f'p{i}'
            connect(self.app, member, [f'd{i}'])
            for v in range(3):
                visit(self.app, f'v{i}_{v}', f'd{i}', activity, WEEKS[v % 2])
        self.record = self.app.execute('segments.analyze', dict(id='s1', k=2))
        self.cluster_of = {}
        for cluster in self.record['clusters']:
            for member in cluster['members']:
                self.cluster_of[member] = cluster['cluster']
        self.app.execute('audiences.save', dict(id='a1', name='赛事关注人群',
                                                members=['p0', 'p2'], reason='人工报名名单'))

    def test_migration_preview_is_read_only_and_consent_aware(self):
        cluster = self.cluster_of['p0']
        self.app.execute('outreach.queue', dict(id='m1', audience='a1', content='赛事提醒'))
        self.app.execute('identity.withdraw', dict(id='p1'))
        before = self.app.store.snapshot()
        preview = self.app.execute('segments.migration',
                                   dict(audience='a1', segmentation='s1', cluster=cluster))
        self.assertEqual(self.app.store.snapshot(), before)
        self.assertEqual(preview['blocked'], ['p1'])
        self.assertEqual(preview['proposed_members'], ['p0'])
        self.assertEqual(preview['kept'], ['p0'])
        self.assertEqual(preview['added'], [])
        self.assertEqual(preview['removed'], ['p2'])
        self.assertEqual(preview['queued_mailings'], ['m1'])
        self.assertTrue(preview['stale'])
        self.assertTrue(preview['warnings'])

    def test_replacement_chain_keeps_legal_scope_consistent(self):
        cluster = self.cluster_of['p0']
        preview = self.app.execute('segments.migration',
                                   dict(audience='a1', segmentation='s1', cluster=cluster))
        self.assertEqual(preview['proposed_members'], ['p0', 'p1'])
        self.app.execute('audiences.replace', dict(id='a1', members=preview['proposed_members'],
                                                   reason='按分群迁移'))
        self.app.execute('outreach.queue', dict(id='m2', audience='a1', content='新内容'))
        self.app.execute('identity.withdraw', dict(id='p0'))
        mailing = self.app.execute('outreach.send', dict(id='m2'))
        self.assertEqual(mailing['sent'], ['p1'])
        deliveries = self.app.store.snapshot()['delivery']
        self.assertEqual([d['id'] for d in deliveries], ['m2:p1'])
        again = self.app.execute('outreach.send', dict(id='m2'))
        self.assertEqual(again['sent'], ['p1'])
        self.assertEqual(len(self.app.store.snapshot()['delivery']), 1)

    def test_delivery_keeps_only_necessary_credentials(self):
        self.app.execute('outreach.queue', dict(id='m1', audience='a1', content='赛事提醒'))
        self.app.execute('outreach.send', dict(id='m1'))
        deliveries = self.app.store.snapshot()['delivery']
        self.assertTrue(deliveries)
        for delivery in deliveries:
            self.assertLessEqual(set(delivery), {'id', 'revision', 'mailing', 'member'})

    def test_migration_rejects_unknown_cluster(self):
        with self.assertRaises(ValueError):
            self.app.execute('segments.migration',
                             dict(audience='a1', segmentation='s1', cluster=99))


class StatusTests(SegmentsTestCase):
    def setUp(self):
        self.app = Application()
        build_groups(self.app, sizes=(3, 3), activities=('match', 'mall'), prefix='s')
        self.app.execute('segments.analyze', dict(id='s1', k=2))

    def test_fresh_segmentation_is_not_stale(self):
        status = self.app.execute('segments.status', dict(id='s1'))
        self.assertFalse(status['stale'])
        self.assertEqual(status['members_withdrawn'], [])

    def test_identity_and_visit_changes_mark_segmentation_stale(self):
        connect(self.app, 'late', ['d_late'])
        status = self.app.execute('segments.status', dict(id='s1'))
        self.assertTrue(status['stale'])
        self.assertEqual(status['new_identities_since'], 1)
        self.assertEqual(status['identity_changes_since'], 1)
        self.app.execute('identity.split', dict(id='s0_0', devices=['d_s0_0']))
        visit(self.app, 'v_late', 'd_s0_1', 'match', WEEKS[2])
        status = self.app.execute('segments.status', dict(id='s1'))
        self.assertEqual(status['identity_changes_since'], 2)
        self.assertEqual(status['visit_changes_since'], 1)

    def test_withdrawn_members_are_listed(self):
        self.app.execute('identity.withdraw', dict(id='s0_0'))
        status = self.app.execute('segments.status', dict(id='s1'))
        self.assertEqual(status['members_withdrawn'], ['s0_0'])
        self.assertTrue(status['stale'])


class RelateTests(SegmentsTestCase):
    def setUp(self):
        self.app = Application()
        self.groups = build_groups(self.app, sizes=(4, 4, 4))
        self.record = self.app.execute('segments.analyze', dict(id='s1', k=3))
        self.cluster_of = {}
        for cluster in self.record['clusters']:
            for member in cluster['members']:
                self.cluster_of[member] = cluster['cluster']

    def test_focused_audience_keeps_single_meaning(self):
        self.app.execute('audiences.save', dict(id='a1', name='核心观众',
                                                members=self.groups[0], reason='人工名单'))
        result = self.app.execute('segments.relate', dict(id='s1', audience='a1'))
        self.assertEqual(result['dominant_cluster'], self.cluster_of[self.groups[0][0]])
        self.assertEqual(result['alerts'], [])
        self.assertEqual(sum(result['distribution'].values()), 4)

    def test_scattered_audience_triggers_alert(self):
        members = self.groups[0][:2] + self.groups[1][:2]
        self.app.execute('audiences.save', dict(id='a2', name='混合人群',
                                                members=members, reason='人工名单'))
        result = self.app.execute('segments.relate', dict(id='s1', audience='a2'))
        self.assertIsNone(result['dominant_cluster'])
        self.assertTrue(any('分散' in a for a in result['alerts']))

    def test_withdrawn_members_are_reported_not_covered(self):
        members = list(self.groups[0])
        self.app.execute('audiences.save', dict(id='a3', name='将变动人群',
                                                members=members, reason='人工名单'))
        self.app.execute('identity.withdraw', dict(id=members[0]))
        self.app.execute('segments.analyze', dict(id='s2', k=3))
        result = self.app.execute('segments.relate', dict(id='s2', audience='a3'))
        self.assertEqual(result['not_covered_members'], 1)
        self.assertTrue(any('不在当前分群范围' in a for a in result['alerts']))


class HistoryIntegrationTests(SegmentsTestCase):
    def replay_history(self, app):
        operations = json.loads((Path(__file__).parent.parent / 'data' / 'history.json').read_text())
        app.replay(operations)

    def test_history_segmentation_is_consistent_and_deterministic(self):
        self.app = Application()
        self.replay_history(self.app)
        record = self.app.execute('segments.analyze', dict(id='h1', k=3))
        coverage = record['coverage']
        self.assertEqual(coverage['assigned'] + coverage['unassigned_low_data'],
                         coverage['consenting'])
        self.assertEqual(coverage['consenting'] + coverage['excluded_no_consent'],
                         coverage['identities_total'])
        self.assertEqual(coverage['visits_attributed'] + coverage['visits_ambiguous_device']
                         + coverage['visits_unknown_device'] + coverage['visits_excluded_no_consent'],
                         coverage['visits_total'])
        self.assertEqual(sum(c['size'] for c in record['clusters']), coverage['assigned'])
        other = Application()
        try:
            self.replay_history(other)
            again = other.execute('segments.analyze', dict(id='h1', k=3))
        finally:
            other.store.close()
        self.assertEqual(json.dumps(record, sort_keys=True), json.dumps(again, sort_keys=True))
        explain = self.app.execute('segments.explain', dict(id='h1'))
        self.assertEqual(len(explain['clusters']), 3)
        status = self.app.execute('segments.status', dict(id='h1'))
        self.assertFalse(status['stale'])


if __name__ == '__main__':
    unittest.main()

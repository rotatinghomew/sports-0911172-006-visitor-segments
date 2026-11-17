"""可解释行为分群：在现有到访、许可连接与人工名单之上做离线分析。

归因与许可边界见 docs/分群约定.md：
- 设备当前只连接一个主体时，其到访才归因于该主体；共用设备到访不归因于任何人；
- 无分析许可的主体不参与分群，缺失许可不是一种兴趣；
- 解释只含聚合信息，不展示个人到访轨迹；分群不自动起业务名、不自动替换人群。
"""

from datetime import date

from .store import require
from .clustering import standardize, kmeans, silhouette, match_clusters, dist2

FEATURE_LABELS = {
    'visits_total': '可归因到访次数',
    'active_weeks': '活跃周数',
    'distinct_venues': '到访场馆数',
    'share_match': '比赛类到访占比',
    'share_exercise': '运动类到访占比',
    'share_mall': '商圈类到访占比',
    'share_other': '其他活动占比',
    'spend_per_visit': '平均每访消费(分)',
    'recency_days': '距最近到访天数',
}
DEFAULT_FEATURES = list(FEATURE_LABELS)
DEFAULT_SEED = 117206
ACTIVITIES = ('match', 'exercise', 'mall', 'other')

LOW_DATA_VISITS = 3       # 可归因到访少于此数的成员视为低数据
SMALL_CLUSTER_SIZE = 8    # 小于此规模的群提示小样本
WEAK_PURCHASE_MEMBERS = 3  # 有消费成员少于此数提示偏好证据薄弱
PRESERVED_RATIO = 0.6     # 旧群成员主要进入同一新群视为延续
MERGE_RATIO = 0.5         # 多个旧群主要进入同一新群视为合并
LOST_RATIO = 0.5          # 旧群成员仍可归群比例低于此值视为丢失
DOMINANT_RATIO = 0.6      # 名单成员集中于某群的比例阈值


def _parse_day(at):
    return date.fromisoformat(str(at)[:10])


def _week_of(day):
    return day.isocalendar()[:2]


def _attribute(visits, identities):
    """把到访归因到唯一主体；共用或未知设备的到访不归因于任何人。"""
    owners = {}
    for identity in identities:
        for device in identity['devices']:
            owners.setdefault(device, set()).add(identity['id'])
    consent = {identity['id']: bool(identity['consent']) for identity in identities}
    buckets = {}
    counts = {'used': 0, 'no_consent': 0, 'ambiguous': 0, 'unknown': 0}
    for visit in visits:
        candidates = owners.get(visit['device'])
        if not candidates:
            counts['unknown'] += 1
        elif len(candidates) > 1:
            counts['ambiguous'] += 1
        else:
            owner = next(iter(candidates))
            if consent[owner]:
                buckets.setdefault(owner, []).append(visit)
                counts['used'] += 1
            else:
                counts['no_consent'] += 1
    return buckets, counts


def _features(visits, window_end):
    total = len(visits)
    days = [_parse_day(v['at']) for v in visits]
    activities = {a: 0 for a in ACTIVITIES}
    spend = 0
    venues = set()
    for v in visits:
        activities[v['activity']] += 1
        spend += v['cents']
        venues.add(v['venue'])
    return {
        'visits_total': float(total),
        'active_weeks': float(len({_week_of(d) for d in days})),
        'distinct_venues': float(len(venues)),
        'share_match': activities['match'] / total,
        'share_exercise': activities['exercise'] / total,
        'share_mall': activities['mall'] / total,
        'share_other': activities['other'] / total,
        'spend_per_visit': spend / total,
        'recency_days': float((window_end - max(days)).days),
    }


def _run(member_vectors, names, k, seed):
    members = sorted(member_vectors)
    rows = [[member_vectors[m][n] for n in names] for m in members]
    scaled, _, _ = standardize(rows)
    inertia, assign, centroids = kmeans(scaled, k, seed)
    return members, scaled, inertia, assign, centroids


class Segments:
    def __init__(self, store):
        self.store = store

    def analyze(self, id, k, ks=None, seed=DEFAULT_SEED, features=None, min_visits=1):
        """对当前有许可且可归因到访足够的主体分群，保存不可变分群记录。"""
        require(isinstance(k, int) and not isinstance(k, bool) and k >= 2, '群数须为不小于2的整数')
        require(isinstance(min_visits, int) and not isinstance(min_visits, bool) and min_visits >= 1,
                '最小到访数无效')
        names = list(features) if features is not None else list(DEFAULT_FEATURES)
        require(names and len(names) == len(set(names)) and all(n in FEATURE_LABELS for n in names),
                '特征名无效')
        identities = self.store.all('identity')
        visits = self.store.all('visit')
        require(visits, '没有可用到访数据')
        buckets, counts = _attribute(visits, identities)
        consenting = sorted(i['id'] for i in identities if i['consent'])
        assigned = [m for m in consenting if len(buckets.get(m, ())) >= min_visits]
        assigned_set = set(assigned)
        unassigned = [m for m in consenting if m not in assigned_set]
        require(len(assigned) >= k, '可分组成员不足')
        if ks is None:
            candidate_ks = sorted({x for x in (k - 1, k, k + 1) if 2 <= x <= len(assigned)})
        else:
            candidate_ks = sorted(set(ks) | {k})
            require(all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= len(assigned)
                        for x in candidate_ks), '候选群数无效')
        window_end = max(_parse_day(v['at']) for v in visits)
        vectors = {m: _features(buckets[m], window_end) for m in assigned}
        weeks = sorted({w for m in consenting for v in buckets.get(m, ())
                        for w in [_week_of(_parse_day(v['at']))]})

        candidates = []
        results = {}
        for ck in candidate_ks:
            run = _run(vectors, names, ck, seed)
            members, scaled, inertia, assign, _ = run
            sizes = [assign.count(c) for c in range(ck)]
            candidates.append({
                'k': ck,
                'inertia': round(inertia, 3),
                'silhouette': round(silhouette(scaled, assign, ck), 3),
                'min_cluster_size': min(sizes),
                'stability': self._stability(buckets, assigned, names, min_visits, ck, seed, weeks,
                                             dict(zip(members, assign))),
            })
            results[ck] = run

        members, scaled, _, assign, centroids = results[k]
        index = {m: i for i, m in enumerate(members)}
        clusters = []
        for c in range(k):
            cmembers = sorted(m for m, a in zip(members, assign) if a == c)
            dists = [dist2(scaled[index[m]], centroids[c]) ** 0.5 for m in cmembers]
            clusters.append(self._cluster_record(c, cmembers, vectors, buckets, dists))

        record = {
            'k': k,
            'seed': seed,
            'features': names,
            'min_visits': min_visits,
            'basis': self.store.sequence(),
            'coverage': {
                'identities_total': len(identities),
                'consenting': len(consenting),
                'excluded_no_consent': len(identities) - len(consenting),
                'assigned': len(assigned),
                'unassigned_low_data': len(unassigned),
                'visits_total': len(visits),
                'visits_attributed': counts['used'],
                'visits_excluded_no_consent': counts['no_consent'],
                'visits_ambiguous_device': counts['ambiguous'],
                'visits_unknown_device': counts['unknown'],
            },
            'candidates': candidates,
            'stability': next(c['stability'] for c in candidates if c['k'] == k),
            'newcomers': self._newcomers(buckets, consenting, assigned_set, weeks),
            'population': self._population(vectors),
            'clusters': clusters,
            'unassigned': unassigned,
        }
        return self.store.put('segmentation', id, record, create=True)

    def explain(self, id):
        """聚合级解释：每群差异、覆盖与不确定性，不含个人到访轨迹。"""
        record = self.store.get('segmentation', id)
        total = record['coverage']['assigned']
        means = record['population']['means']
        stds = record['population']['stds']
        clusters = []
        for cluster in record['clusters']:
            size = cluster['size']
            rest_size = total - size
            contrasts = []
            if size and rest_size:
                scored = []
                for n in DEFAULT_FEATURES:
                    rest_mean = (means[n] * total - cluster['profile'][n] * size) / rest_size
                    std = stds[n]
                    if std > 0:
                        z = (cluster['profile'][n] - rest_mean) / std
                        scored.append((abs(z), n, z))
                scored.sort(reverse=True)
                for _, n, z in scored[:3]:
                    contrasts.append({'feature': n, 'label': FEATURE_LABELS[n],
                                      'direction': '偏高' if z > 0 else '偏低', 'z': round(z, 3)})
            summary = '、'.join(f"{c['label']}{c['direction']}" for c in contrasts) or '与其余群差异不明显'
            clusters.append({
                'cluster': cluster['cluster'],
                'size': size,
                'share': round(size / total, 3) if total else 0.0,
                'profile': cluster['profile'],
                'contrasts': contrasts,
                'summary': summary,
                'uncertainty': cluster['uncertainty'],
                'warnings': cluster['warnings'],
            })
        return {
            'segmentation': id,
            'k': record['k'],
            'seed': record['seed'],
            'features': record['features'],
            'feature_labels': {n: FEATURE_LABELS[n] for n in record['features']},
            'coverage': record['coverage'],
            'candidates': record['candidates'],
            'stability': record['stability'],
            'newcomers': record['newcomers'],
            'clusters': clusters,
            'notes': ['解释仅含聚合信息，不含个人到访轨迹',
                      '群编号仅本次分群内有效，跨次对应请使用 segments.correspond'],
        }

    def correspond(self, before, after):
        """比较两次分群的成员对应，识别延续、分裂、合并与丢失。"""
        old = self.store.get('segmentation', before)
        new = self.store.get('segmentation', after)
        old_sets = [set(c['members']) for c in old['clusters']]
        new_sets = [set(c['members']) for c in new['clusters']]
        mapping = match_clusters(old_sets, new_sets)
        matrix = [[len(o & n) for n in new_sets] for o in old_sets]
        best = [max(range(len(new_sets)), key=lambda j: (row[j], -j)) if new_sets else None
                for row in matrix]
        hubs = {}
        for i, j in enumerate(best):
            size = old['clusters'][i]['size']
            if j is not None and size and matrix[i][j] / size >= MERGE_RATIO:
                hubs.setdefault(j, []).append(i)
        merged = {i for js in hubs.values() if len(js) > 1 for i in js}
        clusters = []
        alerts = []
        for i, cluster in enumerate(old['clusters']):
            size = cluster['size']
            remaining = sum(matrix[i])
            ratio = matrix[i][best[i]] / size if size and best[i] is not None else 0.0
            if size == 0:
                kind, note = 'empty', '旧群本为空群'
            elif i in merged:
                kind, note = 'merged', '多个旧群并入同一新群，不能沿用旧群号'
            elif remaining / size < LOST_RATIO:
                kind, note = 'lost', '旧群成员大多已不在新分群范围，对应关系丢失'
            elif mapping.get(i) is not None and ratio >= PRESERVED_RATIO:
                kind, note = 'preserved', None
            else:
                kind, note = 'split', '旧群成员分散到多个新群，不能沿用旧群号'
            if note:
                alerts.append(f"旧群 {cluster['cluster']}：{note}")
            clusters.append({
                'cluster': cluster['cluster'],
                'size': size,
                'kind': kind,
                'best_match': best[i] if size else None,
                'overlap_ratio': round(ratio, 3),
                'remaining_ratio': round(remaining / size, 3) if size else None,
            })
        return {
            'before': before,
            'after': after,
            'matrix': matrix,
            'clusters': clusters,
            'unassigned_before': len(old['unassigned']),
            'unassigned_after': len(new['unassigned']),
            'alerts': alerts,
            'stable': not alerts,
        }

    def relate(self, id, audience):
        """对照人工名单与分群：业务含义是否仍对应单一分群。"""
        record = self.store.get('segmentation', id)
        aud = self.store.get('audience', audience)
        member_cluster = {}
        for cluster in record['clusters']:
            for m in cluster['members']:
                member_cluster[m] = cluster['cluster']
        unassigned = set(record['unassigned'])
        distribution = {}
        low_data = 0
        not_covered = 0
        for m in aud['members']:
            if m in member_cluster:
                distribution[member_cluster[m]] = distribution.get(member_cluster[m], 0) + 1
            elif m in unassigned:
                low_data += 1
            else:
                not_covered += 1
        total = len(aud['members'])
        dominant = None
        if total and distribution:
            top = max(distribution, key=lambda c: (distribution[c], -c))
            if distribution[top] / total >= DOMINANT_RATIO:
                dominant = top
        alerts = []
        if distribution and dominant is None:
            alerts.append('名单成员分散在多个群，业务含义不再对应单一分群')
        if not_covered:
            alerts.append('部分成员已不在当前分群范围（许可撤回或归因变化）')
        if low_data:
            alerts.append('部分成员可归因数据不足，未能入群')
        return {
            'audience': audience,
            'name': aud['name'],
            'segmentation': id,
            'size': total,
            'distribution': {str(c): n for c, n in sorted(distribution.items())},
            'unassigned_members': low_data,
            'not_covered_members': not_covered,
            'dominant_cluster': dominant,
            'alerts': alerts,
        }

    def migration(self, audience, segmentation, cluster):
        """只读预览：用某群成员替换名单前，比较新旧成员迁移与可触达覆盖。"""
        aud = self.store.get('audience', audience)
        record = self.store.get('segmentation', segmentation)
        clusters = {c['cluster']: c for c in record['clusters']}
        require(cluster in clusters, '群编号无效')
        target = clusters[cluster]['members']
        blocked = sorted(m for m in target if not self._consent(m))
        proposed = sorted(m for m in target if self._consent(m))
        current = set(aud['members'])
        new = set(proposed)
        identity_changes = self.store.changes_since('identity', record['basis'])
        visit_changes = self.store.changes_since('visit', record['basis'])
        queued = sorted(m['id'] for m in self.store.all('mailing')
                        if m['audience'] == audience and m['state'] == 'queued')
        warnings = []
        if blocked:
            warnings.append('自分群计算后有成员撤回许可，已从建议名单剔除')
        if identity_changes or visit_changes:
            warnings.append('分群计算后身份连接或到访记录已变化，建议重新分群后再替换')
        if queued:
            warnings.append('该人群存在待发消息，替换不影响已生成的待发名单')
        return {
            'audience': audience,
            'segmentation': segmentation,
            'cluster': cluster,
            'current_size': len(current),
            'proposed_size': len(proposed),
            'kept': sorted(current & new),
            'added': sorted(new - current),
            'removed': sorted(current - new),
            'blocked': blocked,
            'proposed_members': proposed,
            'reachable_now': len(proposed),
            'stale': bool(identity_changes or visit_changes),
            'identity_changes_since': len({c['key'] for c in identity_changes}),
            'visit_changes_since': len(visit_changes),
            'queued_mailings': queued,
            'warnings': warnings,
        }

    def status(self, id):
        """分群漂移检查：连接更正、设备拆分、撤回许可或新到访都会使分群过时。"""
        record = self.store.get('segmentation', id)
        basis = record['basis']
        identity_changes = self.store.changes_since('identity', basis)
        visit_changes = self.store.changes_since('visit', basis)
        members = sorted({m for c in record['clusters'] for m in c['members']}
                         | set(record['unassigned']))
        withdrawn = [m for m in members if not self._consent(m)]
        changed = sorted({c['key'] for c in identity_changes})
        new_identities = sorted({c['key'] for c in identity_changes if c['revision'] == 1})
        stale = bool(identity_changes or visit_changes or withdrawn)
        notes = []
        if withdrawn:
            notes.append('有成员已撤回许可，统计与发送需以当前许可为准')
        if changed:
            notes.append('身份连接已变化（连接、拆分或撤回），归因基础可能改变')
        if visit_changes:
            notes.append('有新到访记录，分群可能已过时')
        if not stale:
            notes.append('自分群计算后相关业务记录未变化')
        return {
            'segmentation': id,
            'basis': basis,
            'stale': stale,
            'identity_changes_since': len(changed),
            'changed_identities': changed,
            'new_identities_since': len(new_identities),
            'visit_changes_since': len(visit_changes),
            'members_withdrawn': withdrawn,
            'notes': notes,
        }

    def _consent(self, member):
        try:
            return bool(self.store.get('identity', member)['consent'])
        except ValueError:
            return False

    def _stability(self, buckets, assigned, names, min_visits, k, seed, weeks, full_assign):
        """前半周训练与全量分群的成员一致率；不足两周或样本不足返回 None。"""
        if len(weeks) < 2:
            return None
        half = set(weeks[:len(weeks) // 2])
        half_visits = {m: [v for v in buckets[m] if _week_of(_parse_day(v['at'])) in half]
                       for m in assigned}
        half_members = [m for m in assigned if len(half_visits[m]) >= min_visits]
        if len(half_members) < k:
            return None
        half_end = max(_parse_day(v['at']) for m in half_members for v in half_visits[m])
        half_vectors = {m: _features(half_visits[m], half_end) for m in half_members}
        members, _, _, assign, _ = _run(half_vectors, names, k, seed)
        half_assign = dict(zip(members, assign))
        common = [m for m in half_members if m in full_assign]
        if not common:
            return None
        old_sets = [{m for m in common if half_assign[m] == c} for c in range(k)]
        new_sets = [{m for m in common if full_assign[m] == c} for c in range(k)]
        mapping = match_clusters(old_sets, new_sets)
        agree = sum(1 for m in common if mapping.get(half_assign[m]) == full_assign[m])
        return {'agreement': round(agree / len(common), 3), 'compared': len(common),
                'weeks_train': len(half), 'weeks_total': len(weeks)}

    def _newcomers(self, buckets, consenting, assigned_set, weeks):
        """首个可归因周落在后半周的许可主体；数据不足者保持未入群。"""
        if len(weeks) < 2:
            return None
        second = set(weeks[len(weeks) // 2:])
        first_week = {}
        for m in consenting:
            member_weeks = [_week_of(_parse_day(v['at'])) for v in buckets.get(m, ())]
            if member_weeks:
                first_week[m] = min(member_weeks)
        newcomers = sorted(m for m, w in first_week.items() if w in second)
        return {
            'count': len(newcomers),
            'assigned': sum(1 for m in newcomers if m in assigned_set),
            'unassigned': sum(1 for m in newcomers if m not in assigned_set),
        }

    def _population(self, vectors):
        means = {}
        stds = {}
        size = len(vectors)
        for name in DEFAULT_FEATURES:
            values = [v[name] for v in vectors.values()]
            mean = sum(values) / size
            variance = sum((x - mean) ** 2 for x in values) / size
            means[name] = round(mean, 6)
            stds[name] = round(variance ** 0.5, 6)
        return {'means': means, 'stds': stds}

    def _cluster_record(self, c, cmembers, vectors, buckets, dists):
        size = len(cmembers)
        if size:
            profile = {n: round(sum(vectors[m][n] for m in cmembers) / size, 3)
                       for n in DEFAULT_FEATURES}
            mean_distance = round(sum(dists) / size, 3)
            max_distance = round(max(dists), 3)
        else:
            profile = {n: None for n in DEFAULT_FEATURES}
            mean_distance = None
            max_distance = None
        low_data = sum(1 for m in cmembers if vectors[m]['visits_total'] < LOW_DATA_VISITS)
        purchase_members = sum(1 for m in cmembers if any(v['cents'] > 0 for v in buckets[m]))
        purchase_visits = sum(1 for m in cmembers for v in buckets[m] if v['cents'] > 0)
        warnings = []
        if not size:
            warnings.append('空群：没有成员被归入该群')
        else:
            if size < SMALL_CLUSTER_SIZE:
                warnings.append('样本量较小的群，解读需谨慎')
            if purchase_visits == 1:
                warnings.append('仅有一次消费记录，偶然购买不能证明稳定偏好')
            elif 0 < purchase_members < WEAK_PURCHASE_MEMBERS:
                warnings.append('有消费记录的成员过少，偏好证据薄弱')
            if low_data / size >= 0.5:
                warnings.append('低数据成员占比高，归类不确定性大')
        return {
            'cluster': c,
            'size': size,
            'members': cmembers,
            'profile': profile,
            'uncertainty': {
                'low_data_members': low_data,
                'mean_distance': mean_distance,
                'max_distance': max_distance,
                'purchase_members': purchase_members,
                'purchase_visits': purchase_visits,
            },
            'warnings': warnings,
        }

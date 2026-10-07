"""Offline checks: budgets, caching, and retrieval wiring. No provider calls."""
import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import photo_poc as app


class PhotoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = patch.multiple(app, STATE=root / 'state.json', LOCK=root / 'lock')
        self.paths.start()

    def tearDown(self):
        self.paths.stop()
        self.temp.cleanup()

    def test_limit_blocks_sixth_request_before_network(self):
        state = {'calls': {'mistral': 5, 'elastic': 5}}
        with patch.object(app, 'urlopen') as network:
            for service in ('mistral', 'elastic'):
                with self.assertRaises(app.DemoError):
                    app.remote(state, service, 'https://example.com', 'test', {})
            network.assert_not_called()

    def test_failed_transmission_counts_and_survives_reload(self):
        state = app.read_state()
        with patch.object(app, 'urlopen', side_effect=TimeoutError):
            with self.assertRaises(app.DemoError):
                app.remote(state, 'mistral', 'https://example.com', 'test', {})
        self.assertEqual(app.read_state()['calls']['mistral'], 1)

    def test_invalid_state_fails_closed(self):
        app.STATE.write_text('{broken')
        with self.assertRaises(app.DemoError):
            app.read_state()

    def test_cached_concurrent_preparation_does_not_call_services(self):
        app.save_state({'calls': {'mistral': 1, 'elastic': 2}, 'hotspots': [
            {**r, 'observation': 'A visible object.'} for r in app.REGIONS]})
        with patch.object(app, 'remote') as network, concurrent.futures.ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda _: app.prepare(), range(8)))
            network.assert_not_called()
        self.assertTrue(all(r['calls'] == {'mistral': 1, 'elastic': 2} for r in results))

    def test_review_preserves_raw_model_observation(self):
        raw = {'id': 'lincoln', 'observation': 'A seated figure with a fountain.'}
        reviewed = app.review_observation(raw)
        self.assertEqual(reviewed['model_observation'], raw['observation'])
        self.assertTrue(reviewed['observation_reviewed'])
        self.assertIn('standing', reviewed['observation'])

    def test_building_detail_rejects_unrelated_transit_hit(self):
        raw = {'id': 'distant-tower', 'observation': 'A pointed tower.', 'matches': [
            {'record': {'id': 'carriages'}, 'score': 1}]}
        reviewed = app.review_observation(raw)
        self.assertFalse(reviewed['matches'][0]['accepted'])
        self.assertEqual(reviewed['matches'][0]['record'], raw['matches'][0]['record'])

    def test_real_pipeline_builds_one_vision_request_and_batched_search(self):
        state = app.read_state()
        observations = [{'id': r['id'], 'observation': 'A visible object.', 'search_query': r['label']} for r in app.REGIONS]
        responses = [
            {'choices': [{'message': {'content': json.dumps({'items': observations})}}], 'model': 'fixture'},
            {'errors': False, 'items': [{}] * len(app.CATALOG)},
            {'responses': [{'hits': {'hits': [{'_source': app.CATALOG[i % len(app.CATALOG)], '_score': 1}]}} for i in range(len(app.REGIONS))]},
        ]
        values = {'MISTRAL_API_KEY': 'test', 'ELASTICSEARCH_API_KEY': 'test', 'ELASTICSEARCH_URL': 'https://example.com'}
        with patch.object(app, 'remote', side_effect=responses) as remote:
            app.analyze(state, values)
            app.retrieve(state, values)
            self.assertEqual(remote.call_count, 3)
            self.assertEqual(remote.call_args_list[0].args[1], 'mistral')
            self.assertIn('/_bulk?', remote.call_args_list[1].args[2])
            search = remote.call_args_list[2]
            self.assertTrue(search.args[2].endswith('/_msearch'))
            self.assertEqual(len(search.args[4].splitlines()), len(app.REGIONS) * 2)
        self.assertEqual(len(state['hotspots']), 10)
        self.assertEqual(state['hotspots'][0]['matches'][0]['record']['id'], 'lincoln')

    def test_expansion_reuses_original_five_and_indexes_no_new_records(self):
        old = [{'id': r['id'], 'observation': 'Original observation.', 'search_query': r['label']} for r in app.REGIONS[:5]]
        new = [{'id': r['id'], 'observation': 'New observation.', 'search_query': r['label']} for r in app.REGIONS[5:]]
        original = [{**r, **o, 'matches': []} for r, o in zip(app.REGIONS[:5], old)]
        state = {'calls': {'mistral': 3, 'elastic': 4}, 'seeded': True, 'observations': old, 'hotspots': original}
        responses = [
            {'choices': [{'message': {'content': json.dumps({'items': new})}}]},
            {'responses': [{'hits': {'hits': []}} for _ in new]},
        ]
        values = {'MISTRAL_API_KEY': 'test', 'ELASTICSEARCH_API_KEY': 'test', 'ELASTICSEARCH_URL': 'https://example.com'}
        with patch.object(app, 'remote', side_effect=responses) as remote:
            app.analyze(state, values)
            app.retrieve(state, values)
            self.assertEqual(remote.call_count, 2)
            self.assertTrue(remote.call_args_list[1].args[2].endswith('/_msearch'))
            self.assertEqual(len(remote.call_args_list[1].args[4].splitlines()), 10)
        self.assertEqual(state['hotspots'][:5], original)
        self.assertEqual(len(state['hotspots']), 10)


if __name__ == '__main__':
    unittest.main()

"""Index and search a Hello, world! document using credentials from .env."""

import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def main():
    config = {}
    for line in Path(__file__).with_name('.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            name, value = line.removeprefix('export ').split('=', 1)
            config[name.strip()] = value.strip().strip('\"\'')
    config.update(os.environ)
    endpoint = config['ELASTICSEARCH_URL'].rstrip('/')
    api_key = config['ELASTICSEARCH_API_KEY']

    def request(method, path, body=None):
        req = Request(
            endpoint + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                'Authorization': f'ApiKey {api_key}',
                'Content-Type': 'application/json',
            },
            method=method,
        )
        with urlopen(req, timeout=45) as response:
            return json.load(response)

    info = request('GET', '/')
    print(f"Connected to Elasticsearch {info['version']['number']}", flush=True)

    document_id = str(uuid4())
    document = {'message': 'Hello, world!'}
    indexed = request(
        'PUT', f'/hello-world/_create/{document_id}?refresh=wait_for', document
    )
    print(f"Indexed document: {indexed['_index']}/{indexed['_id']}", flush=True)

    results = request('POST', '/hello-world/_search', {
        'query': {'bool': {'filter': {'ids': {'values': [document_id]}},
                           'must': {'match': {'message': 'Hello'}}}}
    })
    hits = results['hits']['hits']
    if len(hits) != 1 or hits[0]['_source'] != document:
        raise RuntimeError('Search did not return the document just indexed')
    print(json.dumps(hits[0]['_source'], indent=2))
    print('Hello world succeeded: document indexed and found by search.')


if __name__ == '__main__':
    try:
        main()
    except HTTPError as error:
        print(f'Elasticsearch returned HTTP {error.code}', file=sys.stderr)
        sys.exit(1)
    except (URLError, KeyError, OSError, RuntimeError) as error:
        print(f'Hello world failed: {error}', file=sys.stderr)
        sys.exit(1)

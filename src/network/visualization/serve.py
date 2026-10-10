"""Serve the network viewer with an explicitly selected external data bundle."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    if not all((data / name).is_file() for name in ['nodes.json','build_summary.json']):
        parser.error('--data-dir must contain nodes.json and build_summary.json')
    app = Path(__file__).resolve().parents[3] / 'visualizations/author_network'

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *params, **kwargs):
            super().__init__(*params, directory=str(app), **kwargs)

        def translate_path(self, path):
            path = unquote(urlsplit(path).path)
            if path.startswith('/data/'):
                target = (data / path[len('/data/'):]).resolve()
                if not target.is_relative_to(data):
                    return str(data / '__not_found__')
                return str(target)
            return super().translate_path(path)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

    print(f'Viewer: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

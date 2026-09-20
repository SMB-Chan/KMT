"""Run the optional browser training server (build web/operator first)."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', help='Use 0.0.0.0 behind an authenticated preview proxy')
    parser.add_argument('--port', default=8000, type=int)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run('operator_training.server:app', host=args.host, port=args.port,
                ws_max_size=16384, ws_max_queue=32, workers=1)

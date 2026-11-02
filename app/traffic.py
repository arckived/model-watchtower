"""Send test traffic to the service so the monitoring dashboard has data.

Examples (run inside the api container):
    python -m app.traffic --requests 300
    python -m app.traffic --requests 300 --noise 0.7
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter

from sklearn.datasets import load_digits


def post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code


def add_noise(pixels):
    return [min(16.0, max(0.0, p + random.uniform(-10, 10))) for p in pixels]


def main():
    parser = argparse.ArgumentParser(description="Generate traffic for the Digit Classifier")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between requests")
    parser.add_argument("--noise", type=float, default=0.0, help="Share of images to distort (0 to 1)")
    parser.add_argument("--error-rate", type=float, default=0.05, help="Share of invalid requests (0 to 1)")
    args = parser.parse_args()

    X, _ = load_digits(return_X_y=True)
    statuses = Counter()

    for i in range(args.requests):
        if random.random() < args.error_rate:
            payload = {"pixels": [1.0] * 10}
        else:
            pixels = X[random.randrange(len(X))].tolist()
            if random.random() < args.noise:
                pixels = add_noise(pixels)
            payload = {"pixels": pixels}

        statuses[post(f"{args.url}/predict", payload)] += 1
        if (i + 1) % 50 == 0:
            print(f"Sent {i + 1} of {args.requests} requests")
        time.sleep(args.delay)

    print("Done. Responses by status code:", dict(statuses))


if __name__ == "__main__":
    main()

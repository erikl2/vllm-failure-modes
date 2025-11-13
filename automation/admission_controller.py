#!/usr/bin/env python3
"""
Simple admission controller for vLLM.
Prevents overload by limiting concurrent requests.
"""

import asyncio
import aiohttp
from aiohttp import web
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class RequestMetrics:
    """Metrics for a completed request."""
    start_time: float
    end_time: float
    success: bool
    tokens: Optional[int] = None

class AdmissionController:
    """Simple admission controller for vLLM."""

    def __init__(self,
                 vllm_url: str = "http://localhost:8000",
                 max_concurrent: int = 3,
                 max_queue: int = 10,
                 request_timeout: float = 600.0):

        self.vllm_url = vllm_url
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.request_timeout = request_timeout

        # State
        self.active_requests = 0
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.metrics = deque(maxlen=100)
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Stats
        self.total_requests = 0
        self.rejected_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    async def proxy_request(self, request_data: dict) -> dict:
        """Proxy a request to vLLM with admission control."""

        # Try to acquire semaphore (blocks if max concurrent reached)
        async with self.semaphore:
            self.active_requests += 1
            start_time = time.time()

            try:
                # Forward to vLLM
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.vllm_url}/v1/completions",
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=self.request_timeout)
                    ) as response:

                        result = await response.json()

                        # Record metrics
                        end_time = time.time()
                        tokens = result.get('usage', {}).get('completion_tokens', 0)

                        self.metrics.append(RequestMetrics(
                            start_time=start_time,
                            end_time=end_time,
                            success=True,
                            tokens=tokens
                        ))

                        self.successful_requests += 1
                        return result

            except Exception as e:
                # Record failure
                end_time = time.time()
                self.metrics.append(RequestMetrics(
                    start_time=start_time,
                    end_time=end_time,
                    success=False
                ))

                self.failed_requests += 1
                raise

            finally:
                self.active_requests -= 1

    async def handle_completion(self, request):
        """Handle completion request with admission control."""
        self.total_requests += 1

        # Check if we should admit this request
        if self.active_requests >= self.max_concurrent:
            # Queue is full, reject
            if self.queue.full():
                self.rejected_requests += 1
                return web.json_response(
                    {
                        'error': 'Server overloaded, try again later',
                        'queue_size': self.queue.qsize(),
                        'max_queue': self.max_queue
                    },
                    status=503
                )

        try:
            # Get request data
            request_data = await request.json()

            # Proxy to vLLM
            result = await self.proxy_request(request_data)

            return web.json_response(result)

        except asyncio.TimeoutError:
            return web.json_response(
                {'error': 'Request timeout'},
                status=504
            )
        except Exception as e:
            return web.json_response(
                {'error': str(e)},
                status=500
            )

    async def handle_metrics(self, request):
        """Return controller metrics."""

        # Calculate statistics
        recent_latencies = [
            m.end_time - m.start_time
            for m in self.metrics
            if m.success
        ]

        avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0

        recent_throughputs = []
        if len(self.metrics) >= 2:
            for i in range(1, len(self.metrics)):
                dt = self.metrics[i].start_time - self.metrics[i-1].start_time
                tokens = self.metrics[i].tokens or 0
                if dt > 0:
                    recent_throughputs.append(tokens / dt)

        avg_throughput = sum(recent_throughputs) / len(recent_throughputs) if recent_throughputs else 0

        return web.json_response({
            'config': {
                'max_concurrent': self.max_concurrent,
                'max_queue': self.max_queue,
                'request_timeout': self.request_timeout
            },
            'state': {
                'active_requests': self.active_requests,
                'queue_size': self.queue.qsize()
            },
            'stats': {
                'total_requests': self.total_requests,
                'successful': self.successful_requests,
                'failed': self.failed_requests,
                'rejected': self.rejected_requests,
                'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0
            },
            'performance': {
                'avg_latency_sec': avg_latency,
                'avg_throughput_tok_per_sec': avg_throughput
            }
        })

def main():
    """Run admission controller."""
    import argparse

    parser = argparse.ArgumentParser(description='Admission controller for vLLM')
    parser.add_argument('--port', type=int, default=8001, help='Port to listen on')
    parser.add_argument('--vllm-url', default='http://localhost:8000', help='vLLM URL')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent requests')
    parser.add_argument('--max-queue', type=int, default=10, help='Max queue size')
    parser.add_argument('--request-timeout', type=float, default=600.0, help='Request timeout (sec)')

    args = parser.parse_args()

    # Create controller
    controller = AdmissionController(
        vllm_url=args.vllm_url,
        max_concurrent=args.max_concurrent,
        max_queue=args.max_queue,
        request_timeout=args.request_timeout
    )

    # Create web app
    app = web.Application()
    app.router.add_post('/v1/completions', controller.handle_completion)
    app.router.add_get('/metrics', controller.handle_metrics)

    print(f"Starting admission controller on port {args.port}")
    print(f"  vLLM URL: {args.vllm_url}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print(f"  Max queue: {args.max_queue}")
    print(f"  Request timeout: {args.request_timeout}s")

    # Run
    web.run_app(app, host='0.0.0.0', port=args.port)

if __name__ == "__main__":
    main()

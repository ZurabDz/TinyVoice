import queue
import threading


class PrefetchIterator:
    """Runs data loading and H2D transfer in a background thread."""

    def __init__(self, iterator, transform_fn, buffer_size=4):
        self.iterator = iterator
        self.transform_fn = transform_fn
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self):
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                # Apply transformation (e.g., to_jax) and queue it
                # The JAX array creation here happens in the thread, triggering async H2D.
                try:
                    transformed = self.transform_fn(item)
                    self.queue.put(transformed)
                except Exception as e:
                    print(f"Prefetch worker error: {e}")
                    self.queue.put(None)
                    break
        except StopIteration:
            self.queue.put(None)
        except Exception as e:
            print(f"Prefetch worker iterator error: {e}")
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def close(self):
        self.stop_event.set()
        # Drain queue to allow worker to exit if stuck in put()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=1.0)

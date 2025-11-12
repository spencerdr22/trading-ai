import os

class TradovateAPI:
    """
    Placeholder Tradovate client.
    Implement OAuth + REST endpoints here.
    """
    def __init__(self, client_id=None, client_secret=None, access_token=None):
        self.client_id = client_id or os.getenv("TRADOVATE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("TRADOVATE_CLIENT_SECRET")
        self.access_token = access_token or os.getenv("TRADOVATE_ACCESS_TOKEN")
        if not self.client_id or not self.client_secret or not self.access_token:
            # Be friendly: don't crash, but warn. Live mode should refuse to run without keys.
            self.ready = False
        else:
            self.ready = True

    def place_order(self, order):
        if not self.ready:
            raise RuntimeError("Tradovate credentials not provided. Set env vars before using live.")
        # TODO: implement REST call to Tradovate
        raise NotImplementedError("Place order implementation required for live trading.")

class MockTradovate:
    """
    Mock for local testing that mimics ack/fill.
    """
    def place_order(self, order):
        return {"status": "ACK", "order": order, "filled_price": order.get("price", None)}

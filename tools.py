class TickerTool():
    def __init__(self, symbols):
        self.params = symbols
        self.command = "/ticker"

    async def run(self, symbol: str):
        return f"Queried ticker: {symbol}"
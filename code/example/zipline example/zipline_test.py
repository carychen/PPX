from zipline.api import order, record, symbol


def initialize(context):
    pass


def handle_data(context, data):
    order(symbol('AAPL'), 100)
    record(AAPL=data.current(symbol('AAPL'), 'price'))
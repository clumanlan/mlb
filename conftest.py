def pytest_addoption(parser):
    parser.addoption('--date', action='store', default=None, help='Game date YYYY-MM-DD')

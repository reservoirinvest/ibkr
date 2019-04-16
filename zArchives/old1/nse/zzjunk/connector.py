# Provides connection to IBKR

class Connector:
    """ A connector to Traders Workstation.

    Usage:
       usa_paper = Connector(platform=1, live_or_paper=1, y=1)
       usa_paper = Connector(platform=1, live_or_paper=1, y=1, client=5)
       usa_paper = Connector(platform=1, live_or_paper=1, y=1, client=5, ip = '127.1.1.1')

    Attr:
       platform:      (Integer) 1 = TWS, 2 = IBG
       live_or_paper: (Integer) 1 = Live, 2 = Paper [default=2]
       y: (Integer) 1 = USA, 2 = NSE

    Returns:
        socket: (Integer) Socket number based on dictionary
        client: (Integer) Client number [default=1]
        ip:     (String) [default = '127.0.0.1']
    """

    def __init__(self, platform, live_or_paper=2, y=1, ip='127.0.0.1', client=1):
        """Returns a Connector object
        Args:
           (platform) - Integer: [1 = TWS | 2 = IBG]
           (live_or_paper) - Integer: [1 = Live | 2 = Paper]
           (y) - Integer: [1 = USA | 2 = NSE]
           ip: (Str) <'127.0.0.1>

        Socket logic:
        TWS + Live + USA:  (1, 1, 1) = 7496
        TWS + Paper + USA: (1, 2, 1) = 7497
        IBG + Live + USA:  (2, 1, 1) = 4001
        IBG + Paper + USA: (2, 2, 1) = 4002

        TWS + Live + NSE:  (1, 1, 2) = 7498
        TWS + Paper + NSE: (1, 2, 2) = 7499
        IBG + Live + NSE:  (2, 1, 2) = 4003
        IBG + Paper + NSE: (2, 2, 2) = 4004

        """

        self.platform = platform
        self.live_or_paper = live_or_paper
        self.y = y
        self.ip = ip
        self.client = client

        # Dictionary of sockets
        sockets = {(1, 1, 1): 7496, (1, 2, 1): 7497, (2, 1, 1): 4001, (2, 2, 1): 4002,
                   (1, 1, 2): 7498, (1, 2, 2): 7499, (2, 1, 2): 4003, (2, 2, 2): 4004}

        self.socket = sockets[platform, live_or_paper, y]

# socket_test = Connector(platform=1, live_or_paper=2, y=2)
# print(socket_test.socket)


from ib_insync import IB
import logging


def connect_ibkr(max_retries: int = 5, initial_client_id: int = 200) -> IB:
    ib = IB()
    attempt = 0
    while attempt < max_retries:
        try:
            ib.connect("127.0.0.1", 7497, clientId=initial_client_id)  # 7497 = Paper, 7496 = Live
            logging.info("Connected to IBKR TWS/Gateway")
            return ib
        except Exception as e:
            attempt += 1
            logging.warning(f"Connection attempt {attempt} failed: {e}")
            ib.sleep(10)
    logging.error("Failed to connect to IBKR after all retries")
    return None

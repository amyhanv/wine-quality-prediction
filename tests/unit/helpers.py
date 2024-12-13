from helpers.utilities import setup_logger

def test_setup_logger():
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == 20  # INFO level

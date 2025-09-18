import time
from datetime import timedelta


def get_time_taken(start_time: float) -> str:
    """Calculates elapsed time in HH:MM:SS

    Args:
        start_time (float): start timestamp from time.time()

    Returns:
        str: Elapsed time in HH:MM:SS
    """
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    td = timedelta(seconds=elapsed_seconds)

    # Get the total seconds from the timedelta
    total_seconds = int(td.total_seconds())

    # Calculate hours, minutes, and seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format the output as HH:MM:SS
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return formatted_time

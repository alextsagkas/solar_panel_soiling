def get_time(time: float) -> str:
    """Returns a string of the time in seconds or minutes, depending on the value of `time` (when time is greater than 60 sec, it is converted to minutes, weatherwise it is left as seconds).

    Args:
        time (float): Time in seconds.

    Returns:
        str: Time in seconds or minutes.
    """
    if time >= 60:
        return (f"{(time/60.0):.2f} min")
    else:
        return (f"{(time):.2f} sec")

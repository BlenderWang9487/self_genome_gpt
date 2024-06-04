from datetime import datetime


def get_time_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def check_colab():
    """
    Check if the code is running in Google Colab.

    Returns:
        _type_: _description_
    """
    try:
        from google.colab import userdata

        is_colab = True
    except:
        is_colab = False
    return is_colab

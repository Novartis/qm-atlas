import logging
from multiprocessing import Pool
from multiprocessing.context import TimeoutError
from os import environ

import requests

MATOMO_URL = environ.get("MATOMO_URL", None)
MATOMO_SITE_ID = environ.get("MATOMO_SITE_ID", None)
MATOMO_TIMEOUT = environ.get("MATOMO_TIMEOUT", "100")
USER = environ.get("USER", None)
HOSTNAME = environ.get("HOSTNAME", None)

logger = logging.getLogger(__name__)


def health_check():

    # If no site-id, don't register anything

    if (
        MATOMO_URL is None
        or MATOMO_SITE_ID is None
        or MATOMO_TIMEOUT is None
        or USER is None
        or HOSTNAME is None
    ):

        logger.debug(f"{MATOMO_URL}, {MATOMO_SITE_ID}, {MATOMO_TIMEOUT}, {USER}, {HOSTNAME}")
        return False

    return True


def register_event(module: str, args: str):

    logger.debug("Tracking matomo user action")

    user_agent = "bash"
    url = f"http://qm_atlas/{module}"
    url_ref = f"http://{HOSTNAME}"

    query_params = {
        "idsite": MATOMO_SITE_ID,
        "rec": 1,
        "url": url,
        "urlref": url_ref,
        "ua": user_agent,
        "uid": USER,
        "action_name": args,
        "dimension1": args,
        "_ref": HOSTNAME,
    }

    response = requests.post(
        str(MATOMO_URL),
        params=query_params,
        timeout=int(MATOMO_TIMEOUT),
    )

    logger.debug(f"matomo register: {query_params}")

    return response


def register_event_subprocess(module: str, args: str):
    pool = Pool(processes=1)
    results = pool.apply_async(register_event, args=(module, args))
    return results, pool


def close_tracking(result, pool):
    try:
        response = result.get(timeout=2)
        logger.debug(f"matomo tracking: {response}")
    except TimeoutError:
        logger.debug("matomo timed out")
    pool.terminate()

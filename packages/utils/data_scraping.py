import io
import os
import time
from pathlib import Path
from typing import List, Set

import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By


def download_image(
    download_path: Path,
    url: str,
    file_name: str
) -> None:
    """Download an image from a url and save it to download_path/file_name. The image is converted 
    to RGB color format and JPEG file format.

    Args:
        download_path (Path): Path to download the image to.
        url (str): Url of the image to download.
        file_name (str): Name of the file to save the image to.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for 4xx and 5xx HTTP status codes

        os.makedirs(download_path, exist_ok=True)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')

        file_path = os.path.join(download_path, file_name)
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG")

        print(f"[INFO] Image {file_name} downloaded successfully.")

    except requests.exceptions.HTTPError as http_err:
        print(f"[ERROR] HTTP error occurred: {http_err}")

    except requests.exceptions.RequestException as req_err:
        print(f"[ERROR] Request error occurred: {req_err}")

    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e).capitalize()}")


def get_image_urls(
        url: str,
        delay: float,
        max_images: int
) -> Set[str]:
    """Get max_images number of image urls from the given google images url. 

    The delay is the number of seconds to wait so as to allow the images to load. It is used
    while scrolling down the page or clicking on a button.

    Args:
        url (str): Google images url.
        delay (float): Delay in seconds.
        max_images (int): Maximum number of images to scrape.

    Returns:
        Set[str]: Set of image urls.
    """

    print(
        f"[INFO] Scraping {max_images} images from {url}, "
        f"delaying {delay} seconds between scrolls. "
    )

    wd = webdriver.Chrome()
    wd.get(url)

    # Click Reject all button
    _press_button(
        wd=wd,
        button_text="Απόρριψη όλων",
        class_value="Nc7WLe",
        delay=delay,
    )

    image_urls = set()

    while len(image_urls) < max_images:
        _scroll_down(wd, delay)

        thumbnails = wd.find_elements(
            by=By.CLASS_NAME,
            value="Q4LuWd"
        )

        _add_image_urls_to_Set(
            thumbnails=thumbnails,
            wd=wd,
            delay=delay,
            max_images=max_images,
            image_urls=image_urls
        )

    wd.quit()

    return image_urls


def _press_button(
        wd: WebDriver,
        button_text: str,
        class_value: str,
        delay: float,
) -> None:
    """Press a button with the given text and class value, if it exists. Then wait for delay
    seconds.

    Args:
        wd (WebDriver): Chrome webdriver.
        button_text (str): Text of the button to press.
        class_value (str): Class value of the button to press.
        delay (float): Delay in seconds after pressing the button.
    """
    buttons = wd.find_elements(
        by=By.CLASS_NAME,
        value=class_value,
    )
    for button in buttons:
        if button.text == button_text:
            button.click()
            time.sleep(delay)
            break


def _scroll_down(
    wd: WebDriver,
    delay: float,
) -> None:
    """Scroll down the page to load more images. Press the "Show more results" button if it exists.

    After scrolling down, or pressing the button, wait for delay seconds.

    Args:
        wd (_type_): Chrome webdriver.
        delay (_type_): Delay in seconds after scrolling down.
    """
    wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(delay)
    _press_button(
        wd=wd,
        button_text="Show more results",
        class_value="LZ4I",
        delay=delay,
    )


def _add_image_urls_to_Set(
    thumbnails: List,
    wd: WebDriver,
    delay: float,
    max_images: int,
    image_urls: Set[str]
) -> None:
    """Add image urls to the given set. The image urls are extracted from the given thumbnails.
    If the number of image urls in the set is greater than or equal to max_images, then it returns.
    The delay is the number of seconds to wait so as to allow the images to load.

    Precisely, it clicks on each thumbnail to display the big image. Then it finds the urls of the 
    big image (the one that is displayed after interaction) and adds its url to the set. This is 
    done to get the highest resolution image and not the dummy one that is displayed on the 
    thumbnail.

    Args:
        thumbnails (List): List of google images thumbnails.
        wd (WebDriver): Chrome webdriver.
        delay (float): Delay in seconds.
        max_images (int): Maximum number of images to scrape.
        image_urls (Set[str]): Set of image urls.
    """
    for thumbnail in thumbnails:
        # Click on thumbnail to display big image
        try:
            thumbnail.click()
            time.sleep(delay)
        except Exception:
            continue

        # Find big image (the one that is displayed after interaction)
        images = wd.find_elements(
            by=By.CLASS_NAME,
            value="r48jcc"
        )

        for image in images:
            image_src = image.get_attribute("src")

            if image_src is not None and "http" in image_src:  # type: ignore
                image_urls.add(image_src)

        if len(image_urls) >= max_images:
            break

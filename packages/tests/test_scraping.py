from typing import List

from packages.utils.configuration import download_dir
from packages.utils.data_scraping import download_image, get_image_urls


def test_scraping(
    url: str,
    max_images: int,
    timestamp_list: List[str],
    delay: float = 0.8,
    **kwargs,
) -> None:
    # TODO: Add docstring

    image_urls = get_image_urls(
        url=url,
        delay=delay,
        max_images=max_images,
    )

    timestamp_string = "_".join(timestamp_list)

    for i, url in enumerate(image_urls):
        download_image(
            download_path=download_dir,
            url=url,
            file_name=f"{timestamp_string}_{i}.jpg")

import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def extract_data(zip_url, data_dir, project_name):
    """
    Extracts data from `zip_url` to the location identified by `data_dir` parameters.

    Parameters
    ----------
    zip_url: string
        Indicates the external url from where the data is downloaded
    data_dir: string
        Indicates the path to the directory where the data should be saved.
    Returns
    -------

    """
    if not os.path.exists(os.path.join(data_dir, project_name)):
        os.makedirs(data_dir)
        print(f"Created new directory {data_dir}")

        with urlopen(zip_url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(data_dir)
        print(f"Downloaded and unzipped data to the location {data_dir}")
    else:
        print(
            "Directory already exists at the location "
            f"{os.path.join(data_dir, project_name)}"
        )

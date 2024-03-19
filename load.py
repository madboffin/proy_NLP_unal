import os

import pandas as pd
import gdown


def load_data(download_data: bool = False) -> pd.DataFrame:
    file_id = "1D31Z6sUCVUynRyimxs2n8rrULnlwhP9l"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "data/jigsaw_data.zip"
    if download_data:
        if not os.path.exists("data"):
            os.makedirs("data")
        gdown.download(url, output)

    df = pd.read_csv(output, dtype=str)
    print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df

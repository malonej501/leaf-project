import os
import pandas as pd
import re
import requests
import cv2
from pynput.keyboard import Listener
from PIL import Image, ImageTk
import tkinter as tk
import io
import mimetypes
from google_images_search import GoogleImagesSearch
from duckduckgo_search import DDGS
from selenium import webdriver
import DuckDuckGoImages as ddgi
import traceback
import shutil

wd = "leaf_imgs"
file = "Wilf_sample_20-02-24_Janssens_intersect.csv"

startfrom = 2380  # which leaf index to start labelling from

# Google search engine initialisation
api_key = "REMOVED FOR SECURITY REASONS - SEE LOCAL MACHINE"  # API key for leaf-project tied to malonej501@gmail.com
cx = "25948abc932774194"  # search engine id for leaf-finder within leaf-project tied to malonej501@gmail.com


def get_filepaths(species_full):
    filenames_expected = species_full["Filename"].tolist()
    Wilf_sample_filepaths = []
    for dirpath, dirnames, filenames in os.walk(wd):
        # print(filenames)
        for filename in filenames:
            if filename.endswith(".jpg"):
                filename_without_jpg = filename.rstrip(".jpg")
                if filename_without_jpg in filenames_expected:
                    Wilf_sample_filepaths.append(os.path.join(dirpath, filename))

    return Wilf_sample_filepaths


def on_press(event, root):
    if event.char == "q":
        root.destroy()
    else:
        try:
            pressed_keys.append(event.char)
            root.destroy()
        except AttributeError:
            pressed_keys.append(str(event))


def search_google_images(query, canvas, root):
    print(f"Query: {query}")
    gis = GoogleImagesSearch(api_key, cx, validate_images=False)

    _search_params = {
        "q": query,
        "num": 10,
        # "fileType": "jpg|gif|png",
        "rights": "cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived",
        "safe": "safeUndefined",  ##
        "imgType": "imgTypeUndefined",  ##
        "imgSize": "imgSizeUndefined",  ##
        "imgDominantColor": "imgDominantColorUndefined",  ##
        "imgColorType": "imgColorTypeUndefined",  ##
    }

    gis.search(search_params=_search_params)

    for index, item in enumerate(gis.results(), start=1):
        link = item.url
        print(f"result {index} {link}")

        try:
            # Attempt to open the image
            response = requests.get(
                link, stream=True, headers={"User-Agent": "Leaf-Search/1.0"}
            )
            response.raise_for_status()

            # Check if the content is a valid image
            image = Image.open(io.BytesIO(response.content))
            image.thumbnail((250, 250))

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Create label with image
            label = tk.Label(
                canvas, text=f"Result {index}", compound=tk.TOP, image=photo
            )
            label.image = photo
            label.grid(row=(index + 1) % 2, column=(index + 1) // 2, sticky="nw")
        except Exception as e:
            print(f"Error processing image {index}: {e}")
    print("Search complete!")


def search_wikimedia_commons(query, canvas):
    base_url = "https://commons.wikimedia.org/w/api.php"

    # Initialize the Firefox browser in headless mode
    driver_path = (
        "/home/jmalone/Documents/Leaf-Project/Software/Firefox-geckodriver/geckodriver"
    )
    os.environ["PATH"] = driver_path
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)

    # params = {
    #     "action": "query",
    #     "format": "json",
    #     "list": "search",
    #     "srsearch": "Arabidopsis_thaliana",
    #     "srnamespace": "6",  # Namespace for File
    # }
    params = {
        "action": "query",
        "format": "json",
        "list": "allimages",
        "aifrom": query,
        "ailimit": 10,
    }

    response_wiki = requests.get(base_url, params)
    response_wiki_data = response_wiki.json()

    # extract image urls from json response
    urls = [
        image["url"]
        for image in response_wiki_data.get("query", {}).get("allimages", [])
    ]

    for index, item in enumerate(urls):
        url = item
        print(f"result {index} {url}")

        # Visit url with selenium to pre-load the image
        driver.get(url)

        # # Determine the expected content type based on the file extension in the URL
        # file_extension = url.split(".")[-1].lower()
        # expected_content_type = mimetypes.guess_type(f"image.{file_extension}")[0]
        # headers = {"Content-Type": expected_content_type}
        # # print(headers)

        response_img = requests.get(url)

        try:
            # Check if the content is a valid image
            # print(response_img.headers.get("Content-Type", ""))
            image = Image.open(io.BytesIO(response_img.content))
            image.thumbnail((250, 250))

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Create label with image
            label = tk.Label(
                canvas, text=f"Result {index}", compound=tk.TOP, image=photo
            )
            label.image = photo
            label.grid(row=(index + 2) % 2, column=(index + 2) // 2, sticky="nw")
        except Exception as e:
            print(f"Error processing image {index}: {e}")
    print("Search complete!")


def search_duckduckgo_images(query, canvas):
    results = []
    with DDGS() as ddgs:
        keywords = query
        ddgs_images_gen = ddgs.images(keywords, max_results=10)
        for r in ddgs_images_gen:
            results.append(r)

    for index, item in enumerate(results):
        link = item["thumbnail"]
        item_name = item["title"]
        print(f"result {index} {link}")

        try:
            # Attempt to open the image
            response = requests.get(link)
            # print(response.headers.get("Content-Type", ""))

            if response.status_code == 200:
                # Check if the content is a valid image
                image = Image.open(io.BytesIO(response.content))
                image.thumbnail((250, 250))

                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)

                # Create label with image
                label = tk.Label(
                    canvas,
                    text=f"Result {index}",
                    compound=tk.TOP,
                    image=photo,
                )
                label.image = photo
                label.grid(row=(index + 2) % 2, column=(index + 2) // 2, sticky="nw")
            else:
                print(
                    f"Failed to download image {index}. Status code: {response.status_code}"
                )
        except Exception:
            traceback.print_exc()
    print("Search complete!")


def download_duckduckgo_images(query):
    if not os.path.exists(wd + f"ddgi_sample/{query}"):
        os.mkdir(wd + f"ddgi_sample/{query}")
        saved_data = []
        results = []
        with DDGS() as ddgs:
            keywords = query
            ddgs_images_gen = ddgs.images(keywords, max_results=10)
            for r in ddgs_images_gen:
                results.append(r)

        for index, item in enumerate(results):
            saved_data.append(item)
            link = item["thumbnail"]
            print(f"result {index} {link}")

            try:
                # Attempt to open the image
                response = requests.get(link)
                # print(response.headers.get("Content-Type", ""))

                if response.status_code == 200:
                    # Check if the content is a valid image
                    image = Image.open(io.BytesIO(response.content))
                    image.thumbnail((250, 250))

                    image.save(wd + f"ddgi_sample/{query}/result_{index}.png")
                else:
                    print(
                        f"Failed to download image {index}. Status code: {response.status_code}"
                    )
            except Exception:
                traceback.print_exc()
        print("Search complete!")
        saved_data_df = pd.DataFrame(saved_data)
        saved_data_df.to_csv(
            wd + f"ddgi_sample/{query}/results_{query}.csv", index=False
        )


def display_downloaded(query, canvas, species_full):
    path = wd + "ddgi_sample/" + query
    pngs = sorted([file for file in os.listdir(path) if file.lower().endswith(".png")])
    for index, filename in enumerate(pngs):
        png = os.path.join(path, filename)

        try:
            # Check if the content is a valid image
            image = Image.open(png)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Create label with image
            label = tk.Label(
                canvas,
                text=filename,
                compound=tk.TOP,
                image=photo,
            )
            label.image = photo
            label.grid(row=(index + 2) % 2, column=(index + 2) // 2, sticky="nw")

        except Exception:
            traceback.print_exc()
    print("Search complete!")


def process_images_old(species_full):
    leafdata = []
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1000, 1000)
    try:
        Wilf_sample_filepaths = check_alldownloaded(species_full)
        for filename in Wilf_sample_filepaths[startfrom:]:
            if filename.endswith(".jpg"):
                print(filename)
                img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
                cv2.imshow("Image", img)
                key = cv2.waitKey(0)

                if key == 113:  # you can exit the app by pressing q
                    break

                leafdata.append((filename.rstrip(".png"), pressed_keys.copy()[0]))
                pressed_keys.clear()
                print(leafdata[-1])

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Ctrl-C detected. Exiting...")

    finally:
        cv2.destroyAllWindows()
        return leafdata


def process_images(species_full):
    leafdata = []

    Wilf_sample_filepaths = sorted(get_filepaths(species_full))
    try:
        # sorted_file_list = sorted(
        #     os.listdir(wd + "Naturalis_eud_sample_Janssens_intersect_21-01-24"),
        #     key=lambda x: int(re.search(r"\d+", x).group()),
        # )
        for filepath in Wilf_sample_filepaths[startfrom:]:
            if filepath.endswith(".jpg"):
                query = filepath.rsplit("/", 1)[-1]  # .rstrip(".jpg")
                query_split = query.split("_")
                species = query_split[1] + "_" + query_split[2]

                root = tk.Tk()
                root.title(f"Image Viewer: {query}")
                root.geometry("1000x1000")

                canvas = tk.Canvas(root)  # , width=1920, height=1080)
                canvas.grid(row=0, column=0, rowspan=6, columnspan=2)

                print(filepath)
                img = Image.open(filepath)
                img_resized = img.resize((600, 900))
                photo = ImageTk.PhotoImage(img_resized)
                label = tk.Label(canvas, image=photo)
                label.image = photo
                label.grid(row=0, column=0, rowspan=3, sticky="nw")

                # search_google_images(query, canvas, root)
                # search_duckduckgo_images(query, canvas)
                # search_duckduckgo_images_alt(query, canvas)
                # search_wikimedia_commons(query, canvas)
                # display_downloaded(query, canvas, species_full)

                root.bind("<Key>", lambda event, arg=root: on_press(event, arg))
                root.mainloop()

                leafdata.append((query, species, pressed_keys.copy()[0]))
                pressed_keys.clear()
                print(leafdata[-1])

    except Exception:
        traceback.print_exc()

    finally:
        return leafdata


if __name__ == "__main__":
    species_full = pd.read_csv(file)
    pressed_keys = []

    with Listener(on_press=on_press) as listener:
        leafdata = process_images(species_full)
        leafdata_df = pd.DataFrame(leafdata, columns=["filename", "species", "shape"])
        print(leafdata_df)
        leafdata_df.to_csv(f"img_labels_{startfrom}-.csv", index=False)

# if __name__ == "__main__":
#     check_alldownloaded()
#     if os.path.isdir(wd + "ddgi_sample"):
#         shutil.rmtree(wd + "ddgi_sample")
#     os.mkdir(wd + "ddgi_sample")

#     sorted_file_list = sorted(
#         os.listdir(wd + "Naturalis_eud_sample_Janssens_intersect_21-01-24"),
#         key=lambda x: int(re.search(r"\d+", x).group()),
#     )
#     for filename in sorted_file_list[startfrom:]:
#         if filename.endswith(".png"):
#             imgpath = wd + os.path.join(
#                 "Naturalis_eud_sample_Janssens_intersect_21-01-24", filename
#             )
#             query = re.sub(r"\d+", "", filename.rstrip(".png"))
#             print(filename)
#             download_duckduckgo_images(query)

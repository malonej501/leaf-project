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

# wd = "sample_eud_21-1-24/"
wd = "jan_zun_nat_ang_26-09-24/"
# file = "Naturalis_eud_sample_Janssens_intersect_21-01-24.csv"
file = "jan_zun_union_nat_genus.csv"

startfrom = 618  # which leaf index to start labelling from

# Google search engine initialisation
api_key = "REMOVED FOR SECURITY REASONS - SEE LOCAL MACHINE"  # API key for leaf-project tied to malonej501@gmail.com
cx = "25948abc932774194"  # search engine id for leaf-finder within leaf-project tied to malonej501@gmail.com


def check_alldownloaded(species_full, herb_dir):
    filenames_expected = [
        f"{value}{index}" for index, value in enumerate(species_full["species"])
    ]
    for i in range(len(filenames_expected)):
        filenames_expected[i] += ".png"
    filenames = os.listdir(herb_dir)
    # filenames = os.listdir("download_imgs")

    missing_imgs = list(set(filenames_expected).difference(filenames))

    print("Missing imgs:")
    if missing_imgs:
        for i in missing_imgs:
            print(i)
        print("ERROR: not all images in sample are present")
        exit()
    else:
        print("None!")


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
    if not os.path.exists(wd + f"ddgi_sample_1/{query}"):
        os.mkdir(wd + f"ddgi_sample_1/{query}")
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

                    image.save(wd + f"ddgi_sample_1/{query}/result_{index}.png")
                else:
                    print(
                        f"Failed to download image {index}. Status code: {response.status_code}"
                    )
            except Exception:
                traceback.print_exc()
        print("Search complete!")
        saved_data_df = pd.DataFrame(saved_data)
        saved_data_df.to_csv(
            wd + f"ddgi_sample_1/{query}/results_{query}.csv", index=False
        )


def display_downloaded(query, canvas, species_full, ddgi_dir):
    # path = wd + "ddgi_sample/" + query
    path = os.path.join(ddgi_dir, query)
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


def process_images_old():
    leafdata = []
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1000, 1000)
    try:
        sorted_file_list = sorted(
            os.listdir(wd + "Naturalis_eud_sample_Janssens_intersect_13-01-24"),
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        for filename in sorted_file_list[startfrom:]:
            if filename.endswith(".png"):
                print(filename)
                imgpath = wd + os.path.join(
                    "Naturalis_eud_sample_Janssens_intersect_13-01-24", filename
                )
                img = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR)
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


def process_images(species_full, herb_dir, ddgi_dir):
    leafdata = []

    try:
        sorted_file_list = sorted(
            os.listdir(herb_dir),
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        for filename in sorted_file_list[startfrom:]:
            if filename.endswith(".png"):
                imgpath = os.path.join(herb_dir, filename)
                query = re.sub(r"\d+", "", filename.rstrip(".png"))

                root = tk.Tk()
                root.title(f"Image Viewer: {query}")
                root.geometry("1900x902")

                canvas = tk.Canvas(root)  # , width=1920, height=1080)
                canvas.grid(row=0, column=0, rowspan=6, columnspan=2)

                img = Image.open(imgpath)
                img_resized = img.resize((600, 900))
                photo = ImageTk.PhotoImage(img_resized)
                label = tk.Label(canvas, image=photo)
                label.image = photo
                label.grid(row=0, column=0, rowspan=3, sticky="nw")

                # search_google_images(query, canvas, root)
                # search_duckduckgo_images(query, canvas)
                # search_duckduckgo_images_alt(query, canvas)
                # search_wikimedia_commons(query, canvas)

                if os.path.exists(ddgi_dir):
                    display_downloaded(query, canvas, species_full, ddgi_dir)

                print(filename)
                root.bind("<Key>", lambda event, arg=root: on_press(event, arg))
                root.mainloop()

                leafdata.append((filename.rstrip(".png"), pressed_keys.copy()[0]))
                pressed_keys.clear()
                print(leafdata[-1])

    except Exception:
        traceback.print_exc()

    finally:
        return leafdata


if __name__ == "__main__":
    species_full = pd.read_csv(wd + file)
    herb_dir = wd + file.split(".")[0]
    # herb_dir = wd
    ddgi_dir = wd + "ddgi_sample_" + file.split(".")[0]
    check_alldownloaded(species_full, herb_dir)
    pressed_keys = []

    with Listener(on_press=on_press) as listener:
        leafdata = process_images(species_full, herb_dir, ddgi_dir)
        leafdata_df = pd.DataFrame(leafdata, columns=["species", "shape"])
        print(leafdata_df)
        leafdata_df.to_csv(wd + f"img_labels.csv", index=False)

# if __name__ == "__main__":
#     species_full = pd.read_csv(wd + file)
#     check_alldownloaded(species_full)

#     ddgi_path = wd + "ddgi_sample_" + file.split(".")[0]
#     print(ddgi_path)

#     if os.path.isdir(wd + "ddgi_sample_1"):
#         shutil.rmtree(wd + "ddgi_sample_1")
#     os.mkdir(wd + "ddgi_sample_1")

#     sorted_file_list = sorted(
#         os.listdir(wd + "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept"),
#         # os.listdir("download_imgs"),
#         key=lambda x: int(re.search(r"\d+", x).group()),
#     )
#     for filename in sorted_file_list[startfrom:]:
#         if filename.endswith(".png"):
#             imgpath = wd + os.path.join(
#                 "Naturalis_multimedia_eud_sample_10-09-24_zuntini_intercept", filename
#             )
#             # imgpath = os.path.join("download_imgs", filename)
#             query = re.sub(r"\d+", "", filename.rstrip(".png"))
#             print(filename)
#             download_duckduckgo_images(query)

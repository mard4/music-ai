from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from subprocess import CREATE_NO_WINDOW

#Step 1: Initialise the Selenium driver
def init_driver(headless=False, use_profile=False):
    options = Options()
    options.add_experimental_option("detach", True)
    options.add_argument("--start-maximized")
    options.add_argument("--mute-audio")
    options.add_argument("--lang=en")
    options.add_argument("--disable-gpu")
    
    if headless:
        options.add_argument("--headless=new")
    else:
        options.add_argument("--start-maximized")
        
    if use_profile:
        options.add_argument(r"--user-data-dir=C:\Users\marti\AppData\Local\BraveSoftware\Brave-Browser\User Data")
        options.add_argument(r'--profile-directory=Default')
        
    options.binary_location = BRAVE_PATH
    #prefs = {"profile.managed_default_content_settings.images": 2}
    
    service = Service(executable_path = CHROMEDRIVER_PATH)
    service.creation_flags = CREATE_NO_WINDOW
    
    return webdriver.Chrome(service=service, options=options)

#Step 2: Opening the YouTube page
def open_youtube_page(driver, url):
    print("Opening YouTube page...")
    driver.get(url)


#Step 3: Skip cookie pop-ups
def accept_T_and_C(driver, timeout=2):
    try:
        xpath_btn = "//span[contains(text(),'Reject all')]/ancestor::button[1]"
        btn = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath_btn)))
        driver.execute_script("arguments[0].scrollIntoView(true);", btn)
        ActionChains(driver).move_to_element(btn).perform()
        btn.click()
    except TimeoutException:
        print("No cookie request.")

#Step 4: Display the full description
def click_afficher_plus(driver, timeout=8):
    try:
        description = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.ID, "description"))
        )
        parent_zone = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.ID, "description-inline-expander"))
        )
        driver.execute_script("arguments[0].click();", parent_zone)
    except Exception as e:
        print(f"Error with Parent zone or '...more' button: {e}")

#Step 5: Display the transcription
def click_transcription(driver, timeout=15):
    selectors = [
        (By.CSS_SELECTOR, "button[aria-label='Show transcript']"),
        (By.XPATH, "//button[normalize-space()='Show transcript']"),
        (By.XPATH, "//yt-formatted-string[text()='Show transcript']/ancestor::button[1]")
    ]
    for by, sel in selectors:
        try:
            btn = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((by, sel)))
            ActionChains(driver).move_to_element(btn).perform()
            btn.click()
            time.sleep(0.5)
            driver.execute_script("window.scrollTo(0, 0);")
            return
        except Exception as e:
            print(f"Not found or not clickable with: {by} | {sel} | Erreur: {e}")
    try:
        screenshot_path = f"screenshot_transcription_fail_{int(time.time())}.png"
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved : {screenshot_path}")
    except Exception as screen_e:
        print(f"Unable to take a screenshot: {screen_e}")
    raise RuntimeError("After all the trials, the transcript cant be shown.")


#Step 6: Extract the transcript
def extract_transcript(driver, output_basename="transcript"):
    #Creation of sub-folders if required
    output_path = f"{output_basename}.txt"
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #Wait for the transcript to load
    try:
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "ytd-transcript-segment-renderer yt-formatted-string.segment-text")))
        print("Transcription in progress...")
    except TimeoutException:
        print("Transcript not found after 1min.")
        return
    headers = driver.find_elements(
        By.CSS_SELECTOR, "ytd-transcript-section-header-renderer span.yt-core-attributed-string")
    segments = driver.find_elements(
        By.CSS_SELECTOR, "ytd-transcript-segment-renderer yt-formatted-string.segment-text")
    time.sleep(1)
    transcript_lines = []
    for header in headers:
        title = header.text.strip()
        if title:
            transcript_lines.append(f"# {title}")
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript_lines.append(text)
    #Registration in the right folder
    with open(output_path, "w", encoding="utf-8") as f_txt:
        for line in transcript_lines:
            f_txt.write(line + "\n")


#Step 7: Recover all the video links in a playlist
def video_links(driver):
    """
    Récupère tous les liens des vidéos d'une playlist YouTube ouverte dans le navigateur.
    Retourne une liste d'URL complètes.
    """
    #Wait for videos to be uploaded to the DOM
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-playlist-video-renderer"))
    )
    #Select all video title links in the playlist
    video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-playlist-video-renderer a#video-title")
    links = []
    for elem in video_elements:
        href = elem.get_attribute("href")
        if href and "/watch" in href:
            links.append(href)
    return links


#Step 8: Retrieve the playlist name
def get_playlist_name(driver, idx=1):
    """Récupère le nom de la playlist via la balise <title>."""
    try:
        title = driver.title.strip()
        #File system name clean-up
        name = re.sub(r'[\\/*?:"<>|]', "_", title)
        #Remove the " - YouTube" suffix and any brackets
        name = re.sub(r"\s*-\s*YouTube$", "", name)
        name = re.sub(r"^\(\d+\)\s*", "", name)
        if not name:
            name = f"playlist_{idx}"
        return name
    except Exception as e:
        print(f"Unable to retrieve playlist name : {e}")
        return f"playlist_{idx}"

def get_video_title(driver, idx=1):
    """Retrieves the video title using the <title> tag. Cleans up for the file system."""
    try:
        title = driver.title.strip()
        #File system name clean-up
        name = re.sub(r'[\\/*?:"<>|]', "_", title)
        #Remove the suffix " - YouTube" and any brackets
        name = re.sub(r"\s*-\s*YouTube$", "", name)
        name = re.sub(r"^\(\d+\)\s*", "", name)
        if not name:
            name = f"video_{idx}"
        return name
    except Exception as e:
        print(f"Unable to retrieve the video name: {e}")
        return f"video_{idx}"

#Function to retrieve all the video links on a YouTube channel
def channel_video_links(driver, max_scrolls=500):
    """Get all a channel's video links. Returns a list of tuples (url, title)."""
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    scrolls = 0
    print("Scrolling...")
    while scrolls < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        try:
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.documentElement.scrollHeight") > last_height
            )
        except TimeoutException:
            break
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        scrolls += 1
    #Sélectionner le conteneur principal
    try:
        contents = driver.find_element(By.CSS_SELECTOR, "div#contents")
        video_elements = contents.find_elements(By.CSS_SELECTOR, "a#video-title-link")
    except Exception as e:
        print(f"Error when searching for videos in DOM : {e}")
        video_elements = []
    links_titles = []
    for elem in video_elements:
        href = elem.get_attribute("href")
        title = elem.get_attribute("title") or elem.text
        if href and "/watch" in href:
            links_titles.append((href, title))
    return links_titles

def get_channel_name(driver, idx=1):
    """Retrieves the name of the YouTube channel."""
    try:
        #Using the XPath name next to the profile picture
        elem = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/ytd-app/div[1]/ytd-page-manager/ytd-browse/div[4]/ytd-tabbed-page-header/tp-yt-app-header-layout/div/tp-yt-app-header/div[2]/div/div[2]/yt-page-header-renderer/yt-page-header-view-model/div/div[1]/div/yt-dynamic-text-view-model/h1/span"))
        )
        name = elem.text.strip()
        name = re.sub(r'[\\/*?:"<>|]', "_", name)
        if not name:
            name = f"channel_{idx}"
        return name
    except Exception as e:
        print(f"Unable to retrieve the name of the channel : {e}")
        return f"channel_{idx}"

def transcribe_video(url, idx, output_folder=None):
    driver = init_driver()
    try:
        open_youtube_page(driver, url)
        accept_T_and_C(driver)
        click_afficher_plus(driver)
        click_transcription(driver)
        video_title = get_video_title(driver, idx)
        if output_folder:
            output_basename = os.path.join("Transcript", output_folder, video_title)
        else:
            output_basename = os.path.join("Transcript", video_title)
        extract_transcript(driver, output_basename=output_basename)
    except Exception as e:
        print(f"Error for {url} : {e}")
    finally:
        driver.quit()

def get_all_video_urls_and_folder(driver, url):
    """Retourne la liste des URLs de vidéos et le dossier de sortie (None si pas de dossier spécifique)."""
    if "playlist" in url:
        open_youtube_page(driver, url)
        accept_T_and_C(driver)
        playlist_name = get_playlist_name(driver)
        return video_links(driver), playlist_name
    elif "/@" in url and "/videos" in url:
        open_youtube_page(driver, url)
        accept_T_and_C(driver)
        channel_name = get_channel_name(driver)
        return [href for href, _ in channel_video_links(driver)], channel_name
    else:
        return [url], None

def read_youtube_links_from_file(filepath):
    """Reads links from the Youtube_links.txt file, automatically adding 'https://' if missing."""
    links = []
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return links
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        raw_links = content.replace('\n', ',').split(',')
        for link in raw_links:
            link = link.strip()
            if link and not link.startswith("http"):
                link = "https://" + link
            if link:
                links.append(link)
    return links

if __name__ == "__main__":
    
    CHROMEDRIVER_PATH ="./chromedriver.exe"
    
    BRAVE_PATH= "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

    links_file = "./Youtube_links.txt"
    
    OUTPUT_FOLDER = "./data/youtubeTranscripts"
    youtube_urls = read_youtube_links_from_file(links_file)
    print(f"{len(youtube_urls)} links loaded from {links_file}")

    all_video_jobs = []
    driver = init_driver()
    try:
        for url in youtube_urls:
            video_urls, folder = get_all_video_urls_and_folder(driver, url)
            for idx, video_url in enumerate(video_urls, 1):
                all_video_jobs.append((video_url, idx, folder))
    finally:
        driver.quit()

    print(f"{len(all_video_jobs)} videos to transcribe.")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(transcribe_video, video_url, idx, folder)
            for video_url, idx, folder in all_video_jobs
        ]
        for future in as_completed(futures):
            future.result()

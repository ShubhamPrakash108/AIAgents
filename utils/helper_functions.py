from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchRun

def search_duckduckgo_news(query):
    """
    Docstring for search_duckduckgo_news
    :param query: Description

    This function performs a search on DuckDuckGo and returns the results in a list format. It uses the DuckDuckGoSearchResults tool from the langchain_community library, specifying the output format as "list" and the backend as "news". The function takes a query as input and returns the search results.
    """
    print(f"Searching DuckDuckGo for: {query}...")
    search = DuckDuckGoSearchResults(output_format="list", backend="news")
    results = search.invoke(query)
    
    content = ""
    for i, result in enumerate(results):
        content += f"Source: {i+1} Title: {result['title']}\nSnippet: {result['snippet']}\n\n"

    return content

def web_search_duckduckgo(query):
    """
    Docstring for search_duckduckgo
    :param query: Description

    This function performs a search on DuckDuckGo and returns the results in a list format. It uses the DuckDuckGoSearchRun tool from the langchain_community library, specifying the output format as "list". The function takes a query as input and returns the search results.
    """
    print(f"Searching DuckDuckGo for: {query}...")
    search = DuckDuckGoSearchRun()
    results = search.invoke(query)

    return results

def get_weather(city: str) -> str:
    """Gets current weather for a city using Open-Meteo (free, no API key)."""
    print(f"Fetching weather for {city}...")
    import requests

    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    ).json()

    if not geo.get("results"):
        return f"City '{city}' not found."

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    # Then fetch weather
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
        }
    ).json()

    cw = weather["current_weather"]
    return f"{city}: {cw['temperature']}°C, wind {cw['windspeed']} km/h"


def translate_text(text: str, target_lang: str = "en") -> str:
    print(f"Translating text to {target_lang}...")
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="auto", target=target_lang).translate(text)
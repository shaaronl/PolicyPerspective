import requests
from datetime import datetime

def get_location_from_ip():
    """gets country + city based on ip address"""
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        if data["status"] == "success":
            return data["country"], data["city"]
        else:
            print("failed to get location")
            return None, None
    except Exception as e:
        print(f"error getting location: {e}")
        return None, None

def get_time_of_day():
    """Returns the current time of day as a string."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "morning"
    elif 12 <= current_hour < 18:
        return "afternoon"
    else:
        return "evening"
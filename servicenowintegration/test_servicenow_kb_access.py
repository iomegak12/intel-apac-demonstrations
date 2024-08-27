import os
import requests
from dotenv import load_dotenv

headers = {
    "Content-Type": "application/json"
}


def access_service_now_kbs():
    service_now_base_uri = os.environ["SERVICE_NOW_BASE_URI"]
    user_name = os.environ["SERVICE_NOW_USER"]
    password = os.environ["SERVCIE_NOW_PASSWORD"]
    credentials = (user_name, password)
    service_now_kb_uri = f"{service_now_base_uri}?sysparm_limit=10"
    response = requests.get(service_now_kb_uri,
                            auth=credentials, headers=headers)

    status = response.status_code

    if status == 200:
        print("Authentication is Successful ...")

        response = response.json()
        articles = response["result"]

        for article in articles:
            description = article["short_description"]
            print(f"Description: {description}")
    else:
        print("Authentication Failed!")


if __name__ == "__main__":
    try:
        load_dotenv()

        access_service_now_kbs()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")

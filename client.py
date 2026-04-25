import requests

class PrenatalEnvClient:
    """
    OpenEnv HTTPEnvClient for Niva Prenatal Health Environment.
    Communicates via HTTP - never imports server internals.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, user_id: int = 1) -> dict:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"user_id": user_id}
        )
        return response.json()

    def step(self, action: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/step",
            json=action
        )
        return response.json()

    def state(self) -> dict:
        response = requests.get(f"{self.base_url}/state")
        return response.json()

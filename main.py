"""Main entrypoint for running the Salon AI Onboarding application."""

import uvicorn
from service_testing.app import app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import asyncio
from httpx import AsyncClient
from service_testing.app import app

async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/health")
        print("Health Response:", response.json())

if __name__ == "__main__":
    asyncio.run(test_health())

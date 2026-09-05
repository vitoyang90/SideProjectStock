import unittest

from httpx import ASGITransport, AsyncClient

from api.main import app


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_lists_registered_stock_models(self):
        response = await self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(
            payload["supported_stocks"],
            ["2308", "2317", "2330", "2454", "2881"],
        )

    async def test_stocks_exposes_direction_eligibility(self):
        response = await self.client.get("/api/stocks")

        self.assertEqual(response.status_code, 200)
        stocks = {
            item["stock_id"]: item for item in response.json()["stocks"]
        }
        self.assertFalse(stocks["2454"]["direction_signal_eligible"])
        self.assertTrue(stocks["2317"]["direction_signal_eligible"])
        self.assertFalse(stocks["2330"]["binary_shadow_candidate"])
        self.assertFalse(stocks["2317"]["binary_shadow_candidate"])

    async def test_unknown_stock_returns_client_error(self):
        response = await self.client.post("/api/analyze/9999")

        self.assertEqual(response.status_code, 400)
        self.assertIn("No stock-specific model", response.json()["detail"])

    async def test_binary_shadow_rejects_non_candidate_stock(self):
        response = await self.client.post("/api/analyze/2317/binary-shadow")

        self.assertEqual(response.status_code, 400)
        self.assertIn("No V2.2 binary shadow candidate", response.json()["detail"])

    async def test_binary_shadow_rejects_failed_2330_candidate(self):
        response = await self.client.post(
            "/api/analyze/2330/binary-shadow?as_of=2026-09-05"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("No V2.2 binary shadow candidate", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
